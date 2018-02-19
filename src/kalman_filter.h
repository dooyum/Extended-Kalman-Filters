#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "Eigen/Dense"

class KalmanFilter {
public:

  // state vector
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // state transition matrix
  Eigen::MatrixXd F_;

  // process covariance matrix
  Eigen::MatrixXd Q_;

  // measurement matrix
  Eigen::MatrixXd H_;

  // measurement covariance matrix
  Eigen::MatrixXd R_;

  /**
   * Constructor
   */
  KalmanFilter();

  /**
   * Destructor
   */
  virtual ~KalmanFilter();

  /**
   * Init Initializes Kalman filter
   * @param x_in Initial state
   * @param P_in Initial state covariance
   * @param F_in Transition matrix
   * @param H_in Measurement matrix
   * @param R_in Measurement covariance matrix
   * @param Q_in Process covariance matrix
   */
  void Init(Eigen::VectorXd &x_in, Eigen::MatrixXd &P_in, Eigen::MatrixXd &F_in,
      Eigen::MatrixXd &H_in, Eigen::MatrixXd &R_in, Eigen::MatrixXd &Q_in);

  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   * @param delta_T Time between k and k+1 in s
   */
  void Predict();
  
  /**
   * Updates the state for Lidar measurements.
   * @param z The measurement at k+1
   */
  void UpdateWithLidar(const Eigen::VectorXd &z);

  /**
   * Updates the state for radar measurements.
   * @param z The measurement at k+1
   */
  void UpdateWithRadar(const Eigen::VectorXd &z);
  
private:
  /**
   * Updates the state by using standard Kalman Filter equations
   * @param z The measurement at k+1
   */
  void update_(const Eigen::VectorXd &z, const Eigen::VectorXd &z_pred);
};

#endif /* KALMAN_FILTER_H_ */
