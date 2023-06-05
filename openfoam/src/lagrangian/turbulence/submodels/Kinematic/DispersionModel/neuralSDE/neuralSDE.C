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

\*---------------------------------------------------------------------------*/

#include "neuralSDE.H"
#include "demandDrivenData.H"
#include "fvcGrad.H"
#include "fvm.H"
#include "fvc.H"

#include "turbulenceModel.H"
#include "TurbulenceModel.H"
#include "interpolationCellPoint.H"
#include "LESfilter.H"
#include "LESModel.H"
#include "LESeddyViscosity.H"

#include "uniformDimensionedFields.H" //for getting g
#include "keras_model.H"
//#include "fdeep.hpp"

#include "sdeCovariances.H"
//#include "tracerCovarianceMatricesDev.H"
//#include "tracerCovarianceMatricesDevTensorGij.H"

#include "Pstream.H"

// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //

template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::cmptDivideHW(vector v1, vector v2) const
{
    //should only be used with positive denominators       scalar offset = mag(VSMALL);
    scalar i = v1.x() / v2.x();
    scalar j = v1.y() / v2.y();
    scalar k = v1.z() / v2.z();

    /*scalar i = v1.x() / v2.x();
    scalar j = v1.y() / v2.y();
    scalar k = v1.z() / v2.z();
*/
    return vector(i,j,k);

}

template<class CloudType>
Foam::tensor
Foam::neuralSDE<CloudType>::cmptDivideHW(tensor A, tensor B) const
{
    tensor t_out = Zero;
    forAll(t_out, i)
    {
        t_out[i] = A[i] / B[i];
    }
    return t_out;
}

template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::cmptDivideHW(scalar A, vector B) const
{   
    //should only be used with positive denominators   
    //scalar offset = mag(VSMALL);
    scalar i = A / B.x();
    scalar j = A / B.y();
    scalar k = A / B.z();

    /*scalar i = A / B.x();
    scalar j = A / B.y();
    scalar k = A / B.z();
 */   return vector(i,j,k);
}

template<class CloudType>
Foam::tensor
Foam::neuralSDE<CloudType>::cmptDivideHW(scalar A, tensor B) const
{   
    tensor t_out = Zero;
    forAll(t_out, i)
    {   
        t_out[i] = A / B[i];
    }
    return t_out;
}

template<class CloudType>
Foam::tensor
Foam::neuralSDE<CloudType>::cmptMultiplyHW(tensor A, tensor B, tensor C) const
{
    tensor T_out = Zero;
    forAll(T_out,i)
    {
        T_out[i] = A[i] * B[i] * C[i];
    }
    return T_out;
}

template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::cmptMultiplyHW(vector A, vector B, vector C) const
{
    scalar i = A.x() * B.x() * C.x();
    scalar j = A.y() * B.y() * C.y();
    scalar k = A.z() * B.z() * C.z();
    return vector(i,j,k);
}


template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::cmptExp(vector v) const
{
    scalar i = exp(v.x());
    scalar j = exp(v.y());
    scalar k = exp(v.z());

    return vector(i,j,k);
}

template<class CloudType>
Foam::tensor
Foam::neuralSDE<CloudType>::cmptExp(tensor v) const
{
    tensor T_out = Zero;
    forAll(T_out,i)
    {
        T_out[i] = exp(v[i]);
    }
    return T_out;
}

template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::cmptSqrt(vector v) const
{
    scalar i = sqrt(mag(v.x()));
    scalar j = sqrt(mag(v.y()));
    scalar k = sqrt(mag(v.z()));

    return vector(i,j,k);
}

template<class CloudType>
Foam::tensor
Foam::neuralSDE<CloudType>::cmptSqrt(tensor t) const
{
    forAll(t,i)
    {
        t[i] = sqrt(mag(t[i]));
    }

    return t;
}

template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::cmptSqr(vector v) const
{
    scalar i = sqr(v.x());
    scalar j = sqr(v.y());
    scalar k = sqr(v.z());

    return vector(i,j,k);
}

template<class CloudType>
Foam::tensor
Foam::neuralSDE<CloudType>::cmptSqr(tensor t) const
{
    forAll(t,i)
    {
        t[i] = sqr(mag(t[i]));
    }
    return t;
}


template<class CloudType>
Foam::tensor
Foam::neuralSDE<CloudType>::perpParComponentsToTensor(const scalar perp, const scalar par,
                                                                const tensor rirj, const tensor delta ) const
{
    return ((perp * delta) 
           + ( (par-perp) * rirj));
}


template<class CloudType>
Foam::scalar
Foam::neuralSDE<CloudType>::taup(const scalar rhop, const scalar rhof, const scalar dp,
                                            const vector Up, const vector Us, const scalar muc) const
{
    //scalar Ur = mag(Up-Us);
    //scalar Rep = Ur * rhof * dp / muc;
    //scalar taup = rhop * dp * dp / (18.0 * muc);
    //return taup;
    scalar taup = rhop * dp * dp / (18.0 * muc);
    return taup;
    /*if (Rep == 0)
    {
        //scalar taup = rhop * dp * dp / (18.0 * muc);
        scalar taup = mag(VSMALL); ///rhop * dp * dp / (18.0 * muc);
        return taup;
    }
    else
    {
        scalar Cd = 24.0/Rep;//   *(1.0+0.15*pow(Rep,0.687))/Rep;
        scalar taup = (rhop/rhof) * 4.0 * dp / max(3.0 * Cd * Ur, mag(1.0e-50));
        return taup;
    }*/

}

template<class CloudType>
Foam::scalar 
Foam::neuralSDE<CloudType>::integralCoeffA(const scalar deltaT, const scalar X) const
{
    //scalar offset = mag(VSMALL);
    scalar A = -exp(-deltaT/max(X,VSMALL)) + X*(1.0 - exp(-deltaT/max(X,VSMALL)))/deltaT;
    return A;
}

template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::integralCoeffA(const scalar deltaT, const vector X) const
{
    vector ones = vector(1.,1.,1.);

    //-exp(-dt/X) + [1-exp(dt/X)]*X/dt
    vector A = -cmptExp(cmptDivideHW(-deltaT, X)) 
                        + cmptMultiply(ones - cmptExp(cmptDivideHW(-deltaT, X)),X)/deltaT;
    return A;
}

template<class CloudType>
Foam::scalar 
Foam::neuralSDE<CloudType>::integralCoeffB(const scalar deltaT, const scalar X) const
{
    //scalar offset = mag(VSMALL);   
    // 1.0 - X*[1-exp(-dt/X)]/dt
    scalar B = 1.0 - (X*(1.0-exp(-deltaT/max(X,VSMALL)))/deltaT );
    return B;
}

template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::integralCoeffB(const scalar deltaT, const vector X) const
{
    vector ones = vector(1.,1.,1.);

    //1 - [1-exp(-dt/X)]*X/dt
    vector B = ones 
                - cmptMultiply(ones-cmptExp(cmptDivideHW(-deltaT,X)),X)/deltaT;
                             //cmptDivideHW(deltaT,X)) ;
    return B;
}

template<class CloudType>
Foam::scalar
Foam::neuralSDE<CloudType>::integralCoeffCc(const scalar deltaT, const scalar X, const scalar Y) const
{
    scalar Cc = (Y/(Y-X)) * (exp(-deltaT/max(Y,VSMALL)) - exp(-deltaT/max(X,VSMALL))) ;
    return Cc;
}

template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::integralCoeffCc(const scalar deltaT, const scalar X, const vector Y) const
{
    vector ones = vector(1.,1.,1.);
    vector Cc = cmptMultiply(cmptDivide(Y,Y-(ones*X)), 
                             cmptExp(cmptDivideHW(-deltaT,Y))-(exp(-deltaT/max(X,VSMALL))*ones));
    return Cc;
}

template<class CloudType>
Foam::scalar
Foam::neuralSDE<CloudType>::integralCoeffAc(const scalar deltaT, const scalar X, const scalar Y) const
{
    scalar offset = mag(VSMALL);
    scalar Ac = -exp(-deltaT/max(X,offset)) 
                + ((X+Y)/deltaT) * (1.0-exp(-deltaT/max(X,offset)))
                - (1.0 + Y/deltaT)*integralCoeffCc(deltaT,X,Y);
    return Ac;
}

template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::integralCoeffAc(const scalar deltaT, const scalar X, const vector Y) const
{
    vector ones = vector(1.,1.,1.);
    vector Ac = -exp(-deltaT/max(X,VSMALL))*ones
                + ((ones*X+Y)*(1.0-exp(-deltaT/max(X,VSMALL)))/deltaT)
                - cmptMultiply(ones + (Y/deltaT), integralCoeffCc(deltaT,X,Y));
    return Ac;
}

template<class CloudType>
Foam::scalar
Foam::neuralSDE<CloudType>::integralCoeffBc(const scalar deltaT, const scalar X, const scalar Y) const
{
    scalar offset = mag(VSMALL);
    scalar Bc = 1.0 - ((X+Y)/deltaT)*(1.0-exp(-deltaT/max(X,offset))) 
                + Y*integralCoeffCc(deltaT,X,Y)/deltaT;
    return Bc;
}

template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::integralCoeffBc(const scalar deltaT, const scalar X, const vector Y) const
{
    vector ones = vector(1.,1.,1.);
    vector Bc = ones 
                - ((ones*X+Y)*(1.0-exp(-deltaT/max(X,VSMALL)))/deltaT)
                + cmptMultiply(Y/deltaT, integralCoeffCc(deltaT,X,Y));

    return Bc;
}

template<class CloudType>
Foam::tmp<Foam::volScalarField>
Foam::neuralSDE<CloudType>::kModel() const
{
    const objectRegistry& obr = this->owner().mesh();
    const word turbName =
        IOobject::groupName
        (
            turbulenceModel::propertiesName,
            this->owner().U().group()
        );

    if (obr.foundObject<turbulenceModel>(turbName))
    {
        const turbulenceModel& model =
            obr.lookupObject<turbulenceModel>(turbName);
        return model.k();
    }
    else
    {
        FatalErrorInFunction
            << "Turbulence model not found in mesh database" << nl
            << "Database objects include: " << obr.sortedToc()
            << abort(FatalError);

        return tmp<volScalarField>(nullptr);
    }
}

template<class CloudType>
Foam::tmp<Foam::volScalarField>
Foam::neuralSDE<CloudType>::epsilonModel() const
{
    const objectRegistry& obr = this->owner().mesh();
    const word turbName =
        IOobject::groupName
        (
            turbulenceModel::propertiesName,
            this->owner().U().group()
        );

    if (obr.foundObject<turbulenceModel>(turbName))
    {
        const turbulenceModel& model =
            obr.lookupObject<turbulenceModel>(turbName);
        return model.epsilon();
    }
    else
    {
        FatalErrorInFunction
            << "Turbulence model not found in mesh database" << nl
            << "Database objects include: " << obr.sortedToc()
            << abort(FatalError);

        return tmp<volScalarField>(nullptr);
    }
}

template<class CloudType>
Foam::tmp<Foam::volScalarField>
Foam::neuralSDE<CloudType>::nutModel() const
{
    const objectRegistry& obr = this->owner().mesh();
    const word turbName =
        IOobject::groupName
        (
            turbulenceModel::propertiesName,
            this->owner().U().group()
        );

    if (obr.foundObject<turbulenceModel>(turbName))
    {
        const turbulenceModel& model =
            obr.lookupObject<turbulenceModel>(turbName);
        return model.nut();
    }
    else
    {
        FatalErrorInFunction
            << "Turbulence model not found in mesh database" << nl
            << "Database objects include: " << obr.sortedToc()
            << abort(FatalError);

        return tmp<volScalarField>(nullptr);
    }
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class CloudType>
Foam::neuralSDE<CloudType>::neuralSDE
(
    const dictionary& dict,
    CloudType& owner
)
:
    DispersionRASModel<CloudType>(dict, owner),
    kPtr_(nullptr),
    epsilonPtr_(nullptr),
    nutPtr_(nullptr),
    ownEpsilon_(false),
    ownNut_(false),
    injectedParticles_(),
    //gradUcInterpPtr_(nullptr),
    UcSeenInterpPtr_(nullptr),
    pGradInterpPtr_(nullptr),
    uGradInterpPtr_(nullptr),
    UpTildeInterpPtr_(nullptr),
    UsTildeInterpPtr_(nullptr),
    uLaplacianInterpPtr_(nullptr),
    kInterpPtr_(nullptr),
    epsilonInterpPtr_(nullptr),
    ownK_(false),
    UsTilde_
    (
        IOobject
        (
            "UsTilde",
            this->owner().mesh().time().timeName(),
            this->owner().mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->owner().mesh(),
        dimensionedVector("0", dimLength/dimTime, Zero),
        fixedValueFvPatchField<vector>::typeName
        //zeroGradientFvPatchField<vector>::typeName
    ),   
    UpTilde_
    (
        IOobject
        (
            "UpTilde",
            this->owner().mesh().time().timeName(),
            this->owner().mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->owner().mesh(),
        dimensionedVector("0", dimLength/dimTime, Zero),
        zeroGradientFvPatchField<vector>::typeName
        //fixedValueFvPatchField<vector>::typeName
        //zeroGradientFvPatchField<vector>::typeName
    ),
    UsPredTilde_
    (
        IOobject
        (
            "UsPredTilde",
            this->owner().mesh().time().timeName(),
            this->owner().mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->owner().mesh(),
        dimensionedVector("0", dimLength/dimTime, Zero),
        fixedValueFvPatchField<vector>::typeName
        //zeroGradientFvPatchField<vector>::typeName
    ),
    UpPredTilde_
    (
        IOobject
        (
            "UpPredTilde",
            this->owner().mesh().time().timeName(),
            this->owner().mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->owner().mesh(),
        dimensionedVector("0", dimLength/dimTime, Zero),
        zeroGradientFvPatchField<vector>::typeName
        //fixedValueFvPatchField<vector>::typeName
        //zeroGradientFvPatchField<vector>::typeName
    ),
    UName_(dict.subDict("neuralSDECoeffs").lookupOrDefault<word>("U", "U.air")),
    kName_(dict.subDict("neuralSDECoeffs").lookupOrDefault<word>("k", "k.air")),
    epsilonName_(dict.subDict("neuralSDECoeffs").lookupOrDefault<word>("epsilon", "epsilon.air")),
    initCond_((dict.subDict("neuralSDECoeffs").lookupOrDefault<word>("init", "Uc"))),
    debugParticles_(dict.subDict("neuralSDECoeffs").lookupOrDefault("debugParticles", false)),
    debugEuler_(dict.subDict("neuralSDECoeffs").lookupOrDefault("debugEuler", false)),
    stationary_(dict.subDict("neuralSDECoeffs").lookupOrDefault("stationary", false)),
    pGradTermOn_(dict.subDict("neuralSDECoeffs").lookupOrDefault("pGradTermOn", true)),
    viscousTermOn_(dict.subDict("neuralSDECoeffs").lookupOrDefault("viscousTermOn", true)),
    relMotionTermOn_(dict.subDict("neuralSDECoeffs").lookupOrDefault("relMotionTermOn", true)),
    diffusionTermOn_(dict.subDict("neuralSDECoeffs").lookupOrDefault("diffusionTermOn", true)),
    GMultiplier_(dict.subDict("neuralSDECoeffs").lookupOrDefault("GMultiplier", 1.0)),
    BMultiplier_(dict.subDict("neuralSDECoeffs").lookupOrDefault("BMultiplier", 1.0)),
    BNormIndex_(dict.subDict("neuralSDECoeffs").lookupOrDefault("BNormIndex", 1.0)),
    GNormIndex_(dict.subDict("neuralSDECoeffs").lookupOrDefault("GNormIndex", 1.0)),
    uMultiplier_(dict.subDict("neuralSDECoeffs").lookupOrDefault("uMultiplier", 1.0)),
    timescaleInpMultiplier_(dict.subDict("neuralSDECoeffs").lookupOrDefault("timescaleInpMultiplier", 1.0)),
    meshSizeMultiplier_(dict.subDict("neuralSDECoeffs").lookupOrDefault("meshSizeMultiplier", 1.0)),
    Cinit_(dict.subDict("neuralSDECoeffs").lookupOrDefault("Cinit", 1.0)),
    taupVSMALL_(dict.subDict("neuralSDECoeffs").lookupOrDefault("taupVSMALL", false)),
    csanadyFactorsOn_(dict.subDict("neuralSDECoeffs").lookupOrDefault("csanadyFactorsOn", false)),
    relVelIsUpMinusUs_(dict.subDict("neuralSDECoeffs").lookupOrDefault("relVelIsUpMinusUs", false)),
    DFNFeatures_(
       int( readScalar(dict.subDict("neuralSDECoeffs").lookup("NumberOfFeatures")) )  ),
    DFnnDriftModel_
    (
     "DFkerasDriftParameters.nnet",
     true //false
    ),
    DFnnDiffusionModel_
    (
     "DFkerasDiffusionParameters.nnet",
     true //false
    )

{}

template<class CloudType>
Foam::neuralSDE<CloudType>::neuralSDE
(
    const neuralSDE<CloudType>& dm
)
:
    DispersionRASModel<CloudType>(dm),
    kPtr_(nullptr),
    epsilonPtr_(nullptr),
    nutPtr_(nullptr),
    ownEpsilon_(dm.ownEpsilon_),
    ownNut_(dm.ownNut_),
    injectedParticles_(dm.injectedParticles_),
    //timeScaleModel_(nullptr),
    UcSeenInterpPtr_(nullptr),
    pGradInterpPtr_(nullptr),
    uGradInterpPtr_(nullptr),
    UpTildeInterpPtr_(nullptr),
    UsTildeInterpPtr_(nullptr),
    UpPredTildeInterpPtr_(nullptr),
    UsPredTildeInterpPtr_(nullptr),
    uLaplacianInterpPtr_(nullptr),
    kInterpPtr_(nullptr),
    epsilonInterpPtr_(nullptr),
    ownK_(dm.ownK_),
    UsTilde_(dm.UsTilde_),
    UpTilde_(dm.UpTilde_),
    UsPredTilde_(dm.UsPredTilde_),
    UpPredTilde_(dm.UpPredTilde_),
    UName_(dm.UName_),
    kName_(dm.kName_),
    epsilonName_(dm.epsilonName_),
    initCond_(dm.initCond_),
    debugParticles_(dm.debugParticles_),
    debugEuler_(dm.debugEuler_),
    stationary_(dm.stationary_),
    pGradTermOn_(dm.pGradTermOn_),
    viscousTermOn_(dm.viscousTermOn_),
    relMotionTermOn_(dm.relMotionTermOn_),
    diffusionTermOn_(dm.diffusionTermOn_),
    GMultiplier_(dm.GMultiplier_),
    BMultiplier_(dm.BMultiplier_),
    BNormIndex_(dm.BNormIndex_),
    GNormIndex_(dm.GNormIndex_),
    uMultiplier_(dm.uMultiplier_),
    timescaleInpMultiplier_(dm.timescaleInpMultiplier_),
    meshSizeMultiplier_(dm.meshSizeMultiplier_),
    Cinit_(dm.Cinit_),
    taupVSMALL_(dm.taupVSMALL_),
    csanadyFactorsOn_(dm.csanadyFactorsOn_),
    relVelIsUpMinusUs_(dm.relVelIsUpMinusUs_),
    DFNFeatures_(dm.DFNFeatures_),
    DFnnDriftModel_(dm.DFnnDriftModel_),
    DFnnDiffusionModel_(dm.DFnnDiffusionModel_)
{
    //dm.ownK_ = false;
    if (csanadyFactorsOn_)
    {
        Info << "csanadyFactorsOn " << csanadyFactorsOn_ << " for noVelNorm_diffusionScaledByVelSecondMoment.H " << endl;
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class CloudType>
Foam::neuralSDE<CloudType>::~neuralSDE()
{
    cacheFields(false);
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class CloudType>
void Foam::neuralSDE<CloudType>::cacheFields(const bool store)
{

    const objectRegistry& mesh = this->owner().mesh();
    scalar deltaT = mesh.time().deltaT().value();
    scalar currTime = this->owner().mesh().time().value();

    const volVectorField& Uc = mesh.template
                 lookupObject<volVectorField>(UName_);
    const volScalarField& Pg = mesh.template
            lookupObject<volScalarField>("p");
    const volTensorField gradUc = fvc::grad(Uc);
    const volVectorField lapUc  = fvc::laplacian(Uc);

    const volScalarField& rho = this->owner().rho();
    const uniformDimensionedVectorField& g =
            mesh.lookupObject<uniformDimensionedVectorField>("g");
    volScalarField gh("gh", ( g & this->owner().mesh().C() ));
    volVectorField gradPg = fvc::grad(Pg);//-gh); //-gh-ghRef);
    volVectorField gradPg_0 = fvc::grad(Pg.oldTime());
    const volScalarField& mu = this->owner().mu();
    const volScalarField nu = mu/rho;

    Random& rnd = this->owner().rndGen();

    if (store)
    {
        volScalarField delta = mesh.template
            lookupObject<volScalarField>("delta.air");// * deltaMultiplier_;

        volScalarField k = mesh.template
            lookupObject<volScalarField>(kName_);;
        const volScalarField& epsilon = mesh.template
            lookupObject<volScalarField>(epsilonName_);


        if (not pGradTermOn_)
        {
            Info << "pGradTerm off" << endl;
        }
        if (not viscousTermOn_)
        {
            Info << "viscous off" << endl;
        }
        if (not relMotionTermOn_)
        {
            Info << "relMotionTerm off" << endl;
        }
        if (not diffusionTermOn_)
        {
            Info << "diffusionTerm off" << endl;
        }


        const volVectorField gradPg_0 = fvc::grad(Pg.oldTime());
        const volTensorField gradUc_0 = fvc::grad(Uc.oldTime());
        const volVectorField lapUc_0 = fvc::laplacian(Uc.oldTime());

#       include "solveNeuralSDE.H"

    }
    else
    {
        scalar endTime = this->owner().mesh().time().endTime().value();

        // floating point error gives currTime not less than endTime sometimes on last timestep
        if (currTime+deltaT*0.5 < endTime)
        {   
            volScalarField delta = mesh.template
                lookupObject<volScalarField>("delta.air");
            // map particle properties to mesh. Get all properties at particle position 
#           include "averageLagrangianNeuralSDE.H"
        } 

        uGradInterpPtr_.clear();
        uLaplacianInterpPtr_.clear();
        pGradInterpPtr_.clear();
        UcSeenInterpPtr_.clear();
        UpTildeInterpPtr_.clear();
        UsTildeInterpPtr_.clear();

        UsPredTildeInterpPtr_.clear();
        UpPredTildeInterpPtr_.clear();

        kInterpPtr_.clear();
        epsilonInterpPtr_.clear();

    }	

}

template<class CloudType>
Foam::vector Foam::neuralSDE<CloudType>::update
(
    const scalar dt,
    const label celli,
    const vector& U,
    const vector& Uc,
    vector& UTurb,
    scalar& tTurb
)
{
    return UTurb;
}



// ************************************************************************* //

