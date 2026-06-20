/*
 * CDDynamics.h
 * Critical Damped 2nd order Dynamics
 *
 *  Created on: May 29, 2012
 *      Author: Seungsu KIM
 */

#ifndef CDDYNAMICS_H_
#define CDDYNAMICS_H_


#include "MathLib.h"

#ifdef USE_MATHLIB_NAMESPACE
using namespace MathLib;
#endif

class CDDynamics
{
private :
	MathLib::Vector mTarget;
	MathLib::Vector mTargetVelocity;

	MathLib::Vector mState;
	MathLib::Vector mStateVelocity;
	MathLib::Vector mStateAccel;

	MathLib::Vector mPositionLimits;
	MathLib::Vector mVelocityLimits;
	MathLib::Vector mAccelLimits;

	unsigned int mDim;
	double mWn;
	double mDT;

	double mReachingTime;

public :

	CDDynamics(int dim, double dt, double Wn);

	void SetState(const MathLib::Vector & Position);
	void SetState(const MathLib::Vector & Position, const MathLib::Vector & Velocity);
	void SetTarget(const MathLib::Vector & target);
	void SetTarget(double target[]);
	void SetTarget(const MathLib::Vector & target, double ReachingTime);
	void SetStateTarget(const MathLib::Vector & Position, const MathLib::Vector & Target);
	//void SetTarget(const MathLib::Vector & target, const MathLib::Vector & targetVel);

	void SetDt(double dt);
	void SetWn(double Wn);

	void SetAccelLimits(const MathLib::Vector & velLimits);
	void RemoveAccelLimits(void);
	double GetAccelLimits(unsigned int index);

	void SetVelocityLimits(const MathLib::Vector & velLimits);
	void RemoveVelocityLimits(void);
	double GetVelocityLimits(unsigned int index);

	void SetPositionLimits(const MathLib::Vector & posLimits);
	void RemovePositionLimits(void);


	void GetTarget(MathLib::Vector & target);
	//void GetTarget(MathLib::Vector & target, MathLib::Vector & targetVel);

	void GetState(MathLib::Vector & Position);
	void GetState(double *Position);
	void GetState(MathLib::Vector & Position, MathLib::Vector & Velocity);
	void GetStateAccel(MathLib::Vector & Accel);

	void Update();
	void Update(double dt);
	void Update(double dt, double muxVel);

	double GetReachingTime(double dt, double muxVel);
	double GetTargetTime(void);
};


#endif /* CDDYNAMICS_H_ */
