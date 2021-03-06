#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
  * A helper method to calculate Jacobians.
  */
  MatrixXd CalculateJacobian(const VectorXd& x_state);
  
  /**
   * Converts cartesian coordinates to polar coordinates.
   * @param x cartesian coordinates
   */
  VectorXd CartesianToPolar(const VectorXd &x);
  
  /**
   * Converts polar coordinates to cartesian coordinates.
   * @param x polar coordinates
   */
  VectorXd PolarToCartesian(const VectorXd &x);
  
  /**
   * Normalizes the phi value of polar coordinates to be between pi and -pi.
   * @param x polar coordinates
   */
  VectorXd NormalizeRadians(const VectorXd &x);
};

#endif /* TOOLS_H_ */
