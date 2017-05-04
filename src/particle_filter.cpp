/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	// x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	num_particles = 100;
	
	// initialize all particles
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	for(int i = 0; i < num_particles; i++) {
		struct Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta  = dist_theta(gen);
		p.weight = 1;
		
		particles.push_back(p);
		weights.push_back(p.weight);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	std::vector<Particle>::iterator itr;
	
	if(yaw_rate == 0) {
		for(itr = particles.begin(); itr != particles.end(); ++itr) {
			(*itr).x += velocity*delta_t*cos((*itr).theta) + noise_x(gen);
			(*itr).y += velocity*delta_t*sin((*itr).theta) + noise_y(gen);
		}
	} else {
		for(itr = particles.begin(); itr != particles.end(); ++itr) {
			double theta0 = (*itr).theta;
			(*itr).x += velocity*(sin(theta0+yaw_rate*delta_t)-sin(theta0))/yaw_rate;
			(*itr).x += noise_x(gen);
			(*itr).y += velocity*(cos(theta0)-cos(theta0+yaw_rate*delta_t))/yaw_rate;
			(*itr).y += noise_y(gen);
			(*itr).theta += yaw_rate*delta_t + noise_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	// observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	std::vector<LandmarkObs>::iterator oit;
	
	for(oit = observations.begin(); oit != observations.end(); ++oit) {
		std::vector<LandmarkObs>::iterator mit;
		double min_d = 10000000.0; //DBL_MAX;
		
		for(mit = predicted.begin(); mit != predicted.end(); ++mit) {
			double d = dist((*mit).x, (*mit).y, (*oit).x, (*oit).y);
			if(d < min_d) {
				(*oit).id = (*mit).id;
				min_d = d;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	// more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
	
	// create a landmark observation list from map
	double sx = std_landmark[0];
	double sy = std_landmark[1];
	std::vector<LandmarkObs> landMarks;
	std::vector<Map::single_landmark_s>::iterator mit = map_landmarks.landmark_list.begin();
	while(mit != map_landmarks.landmark_list.end()) {
		LandmarkObs lmo;
		
		lmo.x  = (*mit).x_f;
		lmo.y  = (*mit).y_f;
		lmo.id = (*mit).id_i;
		
		landMarks.push_back(lmo);
		mit++;
	}
	weights.clear();
	
	std::vector<Particle>::iterator itr;
	for(itr = particles.begin(); itr != particles.end(); ++itr) {
		// first convert the observations to global coordinate in respect to this particle
		std::vector<LandmarkObs> gpObs;
		std::vector<LandmarkObs>::iterator oit;

		for(oit = observations.begin(); oit != observations.end(); ++oit) {
			LandmarkObs o;
			double x, y;
			
			o.x = (*oit).x*cos((*itr).theta) - (*oit).y*sin((*itr).theta) + (*itr).x;
			o.y = (*oit).x*sin((*itr).theta) + (*oit).y*cos((*itr).theta) + (*itr).y;
			gpObs.push_back(o);
		}

		// assign landmark ids to observations
		dataAssociation(landMarks, gpObs);
		
		// now calculate weight for this particle
		(*itr).weight = 1.0E+300;

		for(oit = gpObs.begin(); oit != gpObs.end(); ++oit) {
			double w;
			LandmarkObs lm = landMarks[((*oit).id-1)];
			
			double dx = (*oit).x - lm.x;
			double dy = (*oit).y - lm.y;
			
			w = exp(-(dx*dx)/(2*sx*sx) - (dy*dy)/(2*sy*sy))/(2*sx*sy*M_PI);
			if (w == 0) continue;  // number too small
			(*itr).weight *= w;
		}
		weights.push_back((*itr).weight);
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::vector<Particle> psamples;
	random_device rd;
	mt19937 gen(rd());
		
	discrete_distribution<int> d(weights.begin(), weights.end());
	
	int nump = particles.size();
	for(int n = 0; n < nump; n++) {
		int i = d(gen);
		psamples.push_back(particles[i]);
	}
	
	weights.clear();
	particles.clear();
	particles = psamples;  // copy resampled particles
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
