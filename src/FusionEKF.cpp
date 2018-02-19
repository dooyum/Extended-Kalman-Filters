#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;
  
  H_laser_ << 1, 0, 0, 0,
        0, 1, 0, 0;

  noise_ax_ = 9.0;
  noise_ay_ = 9.0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
    */
    // first measurement
    cout << "EKF: " << endl;

    MatrixXd P_ = MatrixXd(4,4);
    P_ << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1000, 0,
          0, 0, 0, 1000;
    
    MatrixXd F_ = MatrixXd(4, 4);
    F_ << 1, 0, 1, 0,
          0, 1, 0, 1,
          0, 0, 1, 0,
          0, 0, 0, 1;
    
    MatrixXd Q_ = MatrixXd(4, 4);
    Q_ << 0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0;
    
    previous_timestamp_ = measurement_pack.timestamp_;
    VectorXd z_ = VectorXd(4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      z_ = tools.PolarToCartesian(measurement_pack.raw_measurements_);
      Hj_ << tools.CalculateJacobian(z_);
      ekf_.Init(z_, P_, F_, Hj_, R_radar_, Q_);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      z_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
      ekf_.Init(z_, P_, F_, H_laser_, R_laser_, Q_);
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
   */
  float elapsed_time = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  
  float elapsed_time_2 = elapsed_time * elapsed_time;
  float elapsed_time_3 = elapsed_time_2 * elapsed_time;
  float elapsed_time_4 = elapsed_time_2 * elapsed_time_2;
  
  // Integrate time into F matrix.
  ekf_.F_(0, 2) = elapsed_time;
  ekf_.F_(1, 3) = elapsed_time;
  
  ekf_.Q_ << elapsed_time_4/4*noise_ax_, 0, elapsed_time_3/2*noise_ax_, 0,
            0, elapsed_time_4/4*noise_ay_, 0, elapsed_time_3/2*noise_ay_,
            elapsed_time_3/2*noise_ax_, 0, elapsed_time_2*noise_ax_, 0,
            0, elapsed_time_3/2*noise_ay_, 0, elapsed_time_2*noise_ay_;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    VectorXd z_ = VectorXd(3);
    z_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], measurement_pack.raw_measurements_[2];
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.UpdateWithRadar(z_);
  } else {
    // Laser updates
    VectorXd z_ = VectorXd(2);
    z_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1];
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.UpdateWithLidar(z_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
