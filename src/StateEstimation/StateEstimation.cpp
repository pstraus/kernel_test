#include <StateEstimation.h>

Eigen::Matrix<float, 6, 6> math::se::cartesianStateTransitionMatrix(double dt)
{
  Eigen::Matrix<float, 6, 6> out;
  out << 1, 0, 0, dt, 0, 0,
      0, 1, 0, 0, dt, 0,
      0, 0, 1, 0, 0, dt,
      0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 1;
  return out;
}
