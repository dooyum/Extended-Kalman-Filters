#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    
    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if(estimations.size() != ground_truth.size()
       || estimations.size() == 0){
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }
    
    //accumulate squared residuals
    for(unsigned int i=0; i < estimations.size(); ++i){
        
        VectorXd residual = estimations[i] - ground_truth[i];
        
        //coefficient-wise multiplication
        residual = residual.array()*residual.array();
        rmse += residual;
    }
    
    //calculate the mean
    rmse = rmse/estimations.size();
    
    //calculate the squared root
    rmse = rmse.array().sqrt();
    
    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    MatrixXd Hj(3,4);
    Hj.fill(0.0);
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);
    
    //pre-compute a set of terms to avoid repeated calculation
    float c1 = px*px+py*py;
    float c2 = sqrt(c1);
    float c3 = (c1*c2);
    
    //check division by zero
    if(fabs(c1) < 0.0001){
        cout << "CalculateJacobian () - Error - Division by Zero" << endl;
        return Hj;
    }
    
    //compute the Jacobian matrix
    Hj << (px/c2), (py/c2), 0, 0,
    -(py/c1), (px/c1), 0, 0,
    py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;
    
    return Hj;
}

VectorXd Tools::CartesianToPolar(const VectorXd &x) {
  VectorXd polar(3);
  polar.fill(0.0);
  float px = x(0);
  float py = x(1);
  float px_2 = px * px;
  float py_2 = py * py;
  float vx = x(2);
  float vy = x(3);
  
  //check division by zero
  if(fabs(px) < 0.0001 || fabs(px_2 + py_2) < 0.0001){
    cout << "CartesianToPolar () - Error - Division by Zero" << endl;
    return polar;
  }
  
  float rho = sqrt(px_2 + py_2);
  float phi = atan2(py,px);
  float rho_dot = (px * vx + py * vy) / sqrt(px_2 + py_2);
  
  polar << rho, phi, rho_dot;
  
  return polar;
}

VectorXd Tools::PolarToCartesian(const VectorXd &x) {
  float rho = x(0);
  float phi = x(1);
  float px = rho * cos(phi);
  float py = rho * sin(phi);
  
  VectorXd cartesian(4);
  cartesian << px, py, 0, 0;
  
  return cartesian;
}

VectorXd Tools::NormalizeRadians(const VectorXd &x) {
  float phi = x(1);
  
  while (phi > M_PI) {
    phi -= 2 * M_PI;
  }
  
  while (phi < -M_PI) {
    phi += 2 * M_PI;
  }

  VectorXd normalizedPolar(3);
  normalizedPolar << x(0), phi, x(2);
  return normalizedPolar;
}

