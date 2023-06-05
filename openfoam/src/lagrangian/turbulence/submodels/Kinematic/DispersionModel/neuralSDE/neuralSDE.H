/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::neuralSDE

Description
    The velocity is perturbed in the direction of -grad(k), with a
    Gaussian random number distribution with variance sigma.
    where sigma is defined below

\*---------------------------------------------------------------------------*/

#ifndef neuralSDE_H
#define neuralSDE_H

#include "DispersionRASModel.H"
#include "volFields.H"
#include "interpolation.H"

#include "LESfilter.H"
#include "simpleFilter.H"
#include "IOdictionary.H"
#include "autoPtr.H"
#include "CloudSubModelBase.H"
#include "runTimeSelectionTables.H"

//#include "IsotropyModel.H"
//#include "TimeScaleModel.H"
#include "keras_model.H"
#include "AveragingMethod.H"
#include "DispersionModel.H"
//#include "CovarianceMatrices.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class neuralSDE Declaration
\*---------------------------------------------------------------------------*/

template<class CloudType>
class neuralSDE
:
    public DispersionRASModel<CloudType>
{

    // Private data

    // Private Member Functions
    vector cmptDivideHW(vector v1, vector v2) const;
    tensor cmptDivideHW(tensor A, tensor B) const;
    vector cmptDivideHW(scalar A, vector B) const;
    tensor cmptDivideHW(scalar A, tensor B) const;

    tensor cmptMultiplyHW(tensor A, tensor B, tensor C) const;
    vector cmptMultiplyHW(vector A, vector B, vector C) const;
    vector cmptExp(vector v) const;
    tensor cmptExp(tensor v) const;
    vector cmptSqrt(vector v) const;
    tensor cmptSqrt(tensor t) const;
    vector cmptSqr(vector v) const;
    tensor cmptSqr(tensor v) const;

    // Covariance matrix components for first order predictor
    vector gammaLowerSqr(const scalar deltaT, 
                         const vector sigma_i, 
                         const vector theta,
                         const vector Ti, 
                         const scalar tau) const;

    vector GammaUpperSqr(const scalar deltaT, 
                         const vector sigma_i,
                         const vector theta,
                         const vector Ti,
                         const scalar tau) const;

    vector OmegaSqr(const scalar deltaT, 
                         const vector sigma_i,
                         const vector theta,
                         const vector Ti,
                         const scalar tau) const;

    vector gammaGamma(const scalar deltaT, 
                         const vector sigma_i,
                         const vector theta,
                         const vector Ti,
                         const scalar tau) const;

    vector gammaOmega(const scalar deltaT, 
                         const vector sigma_i,
                         const vector theta,
                         const vector Ti,
                         const scalar tau) const;

    vector GammaOmega(const scalar deltaT, 
                         const vector sigma_i,
                         const vector theta,
                         const vector Ti,
                         const scalar tau) const;

    // Tensor form of stochastic integrals
    tensor gammaLowerSqr(const scalar deltaT,
                         const tensor sigma_i,
                         const tensor theta,
                         const tensor Ti,
                         const scalar tau) const;

    tensor GammaUpperSqr(const scalar deltaT,
                         const tensor sigma_i,
                         const tensor theta,
                         const tensor Ti,
                         const scalar tau) const;

    tensor OmegaSqr(const scalar deltaT,
                         const tensor sigma_i,
                         const tensor theta,
                         const tensor Ti,
                         const scalar tau) const;

    tensor gammaGamma(const scalar deltaT,
                         const tensor sigma_i,
                         const tensor theta,
                         const tensor Ti,
                         const scalar tau) const;

    tensor gammaOmega(const scalar deltaT,
                         const tensor sigma_i,
                         const tensor theta,
                         const tensor Ti,
                         const scalar tau) const;

    tensor GammaOmega(const scalar deltaT,
                         const tensor sigma_i,
                         const tensor theta,
                         const tensor Ti,
                         const scalar tau) const;


/**/

    // Function to convert perpendicular and parallel scalars to a tensor
    tensor perpParComponentsToTensor(const scalar perp, const scalar par,
                                    const tensor rirj, const tensor delta) const;
    // Return particle timescale
    scalar taup(const scalar rhop, const scalar rhof, const scalar dp,
                const vector Up, const vector Us, const scalar muc) const;
    // Integral coefficients for corrector step of Us model
    scalar integralCoeffA(const scalar deltaT, const scalar X) const;
    vector integralCoeffA(const scalar deltaT, const vector X) const;
    scalar integralCoeffB(const scalar deltaT, const scalar X) const;
    vector integralCoeffB(const scalar deltaT, const vector X) const;
    scalar integralCoeffCc(const scalar deltaT,
                           const scalar X,
                           const scalar Y) const;
    vector integralCoeffCc(const scalar deltaT,
                           const scalar X,
                           const vector Y) const;
    scalar integralCoeffAc(const scalar deltaT, 
                           const scalar X, 
                           const scalar Y) const;
    vector integralCoeffAc(const scalar deltaT,
                           const scalar X,
                           const vector Y) const;
    scalar integralCoeffBc(const scalar deltaT, 
                           const scalar X,  
                           const scalar Y) const;
    vector integralCoeffBc(const scalar deltaT,
                           const scalar X,
                           const vector Y) const;

    tmp<volVectorField> smoothen(volVectorField) const;

        //- Reference to the cloud random number generator
        //Random& rndGen_;
	
    //- Pointer to the turbulence kinetic energy field
    const volScalarField* kPtr_;
	const volScalarField* epsilonPtr_;
    //-AO
    const volScalarField* nutPtr_;
    bool ownEpsilon_;
    bool ownNut_;
	//- List of particle labels 
	wordList injectedParticles_;

	//- Carrier phase velocity interpolator
	autoPtr<interpolation<vector>> UcSeenInterpPtr_;
	//- Carrier phase pressure gradient interpolator
	autoPtr<interpolation<vector>> pGradInterpPtr_;
	autoPtr<interpolation<tensor>> uGradInterpPtr_;
	autoPtr<interpolation<vector>> UpTildeInterpPtr_;
    autoPtr<interpolation<vector>> UsTildeInterpPtr_;
    autoPtr<interpolation<vector>> UpPredTildeInterpPtr_;
    autoPtr<interpolation<vector>> UsPredTildeInterpPtr_;
	autoPtr<interpolation<vector>> uLaplacianInterpPtr_;
    autoPtr<interpolation<scalar>> kInterpPtr_;
    autoPtr<interpolation<scalar>> epsilonInterpPtr_;

    //- Flag that indicates ownership of turbulence k field
    bool ownK_;

    //- Fluid velocity seen by particle
	volVectorField UsTilde_;
	volVectorField UpTilde_; 
    volVectorField UsPredTilde_;
    volVectorField UpPredTilde_;
	//- Filtered velocity terms

    const word UName_;
    const word kName_;
    const word epsilonName_;

    const word initCond_;
    const bool debugParticles_;
    const bool debugEuler_;
    const bool stationary_;

    const bool pGradTermOn_;
    const bool viscousTermOn_;
    const bool relMotionTermOn_;
    const bool diffusionTermOn_;
    const scalar GMultiplier_;
    const scalar BMultiplier_;
    const scalar BNormIndex_;
    const scalar GNormIndex_;
    const scalar uMultiplier_;
    const scalar timescaleInpMultiplier_;
    const scalar meshSizeMultiplier_;
    const scalar Cinit_;
    const bool taupVSMALL_;
    const bool csanadyFactorsOn_;
    const bool relVelIsUpMinusUs_;

    const int DFNFeatures_;
    //Neural network model
    mutable keras::KerasModel DFnnDriftModel_;
    mutable keras::KerasModel DFnnDiffusionModel_;

    // Private Member Functions

        //- Return the k field from the turbulence model
        tmp<volScalarField> kModel() const;
	//- Return the epsilon field from the turbulence model
	tmp<volScalarField> epsilonModel() const;
    //- Return the turbulent viscosity from the turbulence model
    tmp<volScalarField> nutModel() const;

public:

    //- Runtime type information
    TypeName("neuralSDE");


    // Constructors

        //- Construct from components
        neuralSDE(const dictionary& dict, CloudType& owner);

        //- Construct copy
        neuralSDE(const neuralSDE<CloudType>& dm);

        //- Construct and return a clone
        virtual autoPtr<DispersionModel<CloudType>> clone() const
        {
            return autoPtr<DispersionModel<CloudType>>
            (
                new neuralSDE<CloudType>(*this)
            );
        }


    //- Destructor
    virtual ~neuralSDE();


    // Member Functions

	//- Return the carrier phase velocity interpolator
	inline const interpolation<vector>& UcSeenInterp() const;
	inline const interpolation<vector>& pGradInterp() const;
	inline const interpolation<tensor>& uGradInterp() const;
	inline const interpolation<vector>& UpTildeInterp() const;
    inline const interpolation<vector>& UsTildeInterp() const;
    inline const interpolation<vector>& UpPredTildeInterp() const;
    inline const interpolation<vector>& UsPredTildeInterp() const;
	inline const interpolation<vector>& uLaplacianInterp() const;
    inline const interpolation<scalar>& kInterp() const;
    inline const interpolation<scalar>& epsilonInterp() const;

        //- Cache carrier fields
        virtual void cacheFields(const bool store);

        //- Update (disperse particles)
        virtual vector update
        (
            const scalar dt,
            const label celli,
            const vector& U,
            const vector& Uc,
            vector& UTurb,
            scalar& tTurb
        );
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
#include "neuralSDEI.H"

#ifdef NoRepository
    #include "neuralSDE.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
