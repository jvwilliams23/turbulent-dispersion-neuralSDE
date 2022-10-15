/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2013-2014 OpenFOAM Foundation
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

#include "MPPICHWCloud.H"
#include "PackingModel.H"
#include "ParticleStressModel.H"
#include "DampingModel.H"
#include "IsotropyModel.H"
#include "TimeScaleModel.H"
#include "AveragingMethod.H"

// * * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * //

template<class CloudType>
void Foam::MPPICHWCloud<CloudType>::setModels()
{
    packingModel_.reset
    (
        PackingModel<MPPICHWCloud<CloudType> >::New
        (
            this->subModelProperties(),
            *this
        ).ptr()
    );
    dampingModel_.reset
    (
        DampingModel<MPPICHWCloud<CloudType> >::New
        (
            this->subModelProperties(),
            *this
        ).ptr()
    );
    isotropyModel_.reset
    (
        IsotropyModel<MPPICHWCloud<CloudType> >::New
        (
            this->subModelProperties(),
            *this
        ).ptr()
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class CloudType>
Foam::MPPICHWCloud<CloudType>::MPPICHWCloud
(
    const word& cloudName,
    const volScalarField& rho,
    const volVectorField& U,
    const volScalarField& mu,
    const dimensionedVector& g,
//	const scalar epsilon0,
//    const volVectorField& electricField,
//    volScalarField& rhoe,
//    volScalarField& surfCharge,
    bool readFields
)
:
    CloudType(cloudName, rho, U, mu, g, false),
    packingModel_(nullptr),
    dampingModel_(nullptr),
    isotropyModel_(nullptr)//,
/*
    charge_(
        IOobject
        (
            "meanCharge",
            U.mesh().time().timeName(),
            U.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        U.mesh(),
		dimensionedScalar("0",dimless,scalar(0))
    ),
    chargeDef_(
        IOobject
        (
            "chargeDeficit",
            U.mesh().time().timeName(),
            U.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        U.mesh(),
		dimensionedScalar("0",dimless,scalar(0))
    ),
    workFunction_(
        IOobject
        (
            "workFunction",
            U.mesh().time().timeName(),
            U.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        U.mesh(),
		dimensionedScalar("0",dimless,scalar(0))
    ),
    granularTemperature_(
        IOobject
        (
            "granularTemperature",
            U.mesh().time().timeName(),
            U.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        U.mesh(),
		dimensionedScalar("0",dimless,scalar(0))
    ),
    volumeFraction_(
        IOobject
        (
            "volumeFraction",
            U.mesh().time().timeName(),
            U.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        U.mesh(),
		dimensionedScalar("0",dimless,scalar(0))
    ),
    us_(
        IOobject
        (
            "solidVelocity",
            U.mesh().time().timeName(),
            U.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        U.mesh(),
		dimensionedVector("0",dimless,vector(vector::zero))
    ),
    electricField_( electricField ),
	rhoe_(rhoe)	
{
    if (this->solution().steadyState())
    {
        FatalErrorIn
        (
            "Foam::MPPICHWCloud<CloudType>::MPPICHWCloud"
            "("
                "const word&, "
                "const volScalarField&, "
                "const volVectorField&, "
                "const volScalarField&, "
                "const dimensionedVector&, "
                "bool"
            ")"
        )   << "MPPIC modelling not available for steady state calculations"
            << exit(FatalError);
    }
*/
{
    if (this->solution().steadyState())
    {
        FatalErrorInFunction
            << "MPPIC modelling not available for steady state calculations"
            << exit(FatalError);
    }
    if (this->solution().active())
    {
        setModels();

        if (readFields)
        {
            parcelType::readFields(*this);
 			this->deleteLostParticles();
        }
    }
}


template<class CloudType>
Foam::MPPICHWCloud<CloudType>::MPPICHWCloud
(
    MPPICHWCloud<CloudType>& c,
    const word& name
)
:
    CloudType(c, name),
    packingModel_(c.packingModel_->clone()),
    dampingModel_(c.dampingModel_->clone()),
    isotropyModel_(c.isotropyModel_->clone()) //,
/*
	rhoe_(c.rhoe_),
    charge_(
        IOobject
        (
            "meanCharge",
            c.mesh().time().timeName(),
            c.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        c.mesh(),
	dimensionedScalar("0",dimless,scalar(0))
    ),
    chargeDef_(
        IOobject
        (
            "chargeDeficit",
            c.mesh().time().timeName(),
            c.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        c.mesh(),
	dimensionedScalar("0",dimless,scalar(0))
    ),
    workFunction_(
        IOobject
        (
            "workFunction",
            c.mesh().time().timeName(),
            c.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        c.mesh(),
	dimensionedScalar("0",dimless,scalar(0))
    ),
    granularTemperature_(
        IOobject
        (
            "granularTemperature",
            c.mesh().time().timeName(),
            c.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        c.mesh(),
	dimensionedScalar("0",dimless,scalar(0))
    ),
    volumeFraction_(
        IOobject
        (
            "volumeFraction",
            c.mesh().time().timeName(),
            c.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        c.mesh(),
	dimensionedScalar("0",dimless,scalar(0))
    ),
    us_( c.us() ),
    electricField_( c.electricField() )
*/
{}


template<class CloudType>
Foam::MPPICHWCloud<CloudType>::MPPICHWCloud
(
    const fvMesh& mesh,
    const word& name,
    const MPPICHWCloud<CloudType>& c
)
:
    CloudType(mesh, name, c),
    packingModel_(nullptr),
    dampingModel_(nullptr),
    isotropyModel_(nullptr)//,
/*
	rhoe_(c.rhoe_),
    charge_(
        IOobject
        (
            "meanCharge",
            c.mesh().time().timeName(),
            c.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        c.mesh(),
	dimensionedScalar("0",dimless,scalar(0))
    ),
    chargeDef_(
        IOobject
        (
            "chargeDeficit",
            c.mesh().time().timeName(),
            c.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        c.mesh(),
	dimensionedScalar("0",dimless,scalar(0))
    ),
    workFunction_(
        IOobject
        (
            "workFunction",
            c.mesh().time().timeName(),
            c.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        c.mesh(),
	dimensionedScalar("0",dimless,scalar(0))
    ),
    granularTemperature_(
        IOobject
        (
            "granularTemperature",
            c.mesh().time().timeName(),
            c.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        c.mesh(),
	dimensionedScalar("0",dimless,scalar(0))
    ),
    volumeFraction_(
        IOobject
        (
            "volumeFraction",
            c.mesh().time().timeName(),
            c.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        c.mesh(),
	dimensionedScalar("0",dimless,scalar(0))
    ),
    us_(
        IOobject
        (
            "solidVelocity",
            c.mesh().time().timeName(),
            c.mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        c.mesh(),
	dimensionedVector("0",dimless,vector(vector::zero))
    ),
    electricField_( c.electricField() )
*/
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class CloudType>
Foam::MPPICHWCloud<CloudType>::~MPPICHWCloud()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class CloudType>
void Foam::MPPICHWCloud<CloudType>::storeState()
{
    cloudCopyPtr_.reset
    (
        static_cast<MPPICHWCloud<CloudType>*>
        (
            clone(this->name() + "Copy").ptr()
        )
    );
}


template<class CloudType>
void Foam::MPPICHWCloud<CloudType>::restoreState()
{
    this->cloudReset(cloudCopyPtr_());
    cloudCopyPtr_.clear();
}


template<class CloudType>
void Foam::MPPICHWCloud<CloudType>::evolve()
{
    if (this->solution().canEvolve())
    {
		typename parcelType::trackingData td(*this);

        this->solve(*this, td);
    }
}

/*
template<class CloudType>
template<class TrackData>
void Foam::MPPICHWCloud<CloudType>::setChargeCellValues(TrackData& td)
{
    forAllIter(typename CloudType, *this, iter)
    {
        typename CloudType::parcelType& p = iter();
        p.setValuesCalled = true;
    }
}
*/
/*
template<class CloudType>
template<class TrackData>
void Foam::MPPICHWCloud<CloudType>::updateParcelCharge(TrackData& td)
{
    
    forAllIter(typename CloudType, *this, iter)
    {
        typename CloudType::parcelType& p = iter();
		p.updateCharge( td, this->db().time().deltaTValue() );
    }
    
    //Correct charge deficit
    td.updateChargeDeficit(*this);
    
    // add the interpolated deficit to particle charge
    forAllIter(typename CloudType, *this, iter)
    {
        typename CloudType::parcelType& p = iter();
        const tetIndices tetIs = p.currentTetIndices();
		scalar& pcharge = p.charge();
        
	//const scalar flux =     td.chargeDefInterp().interpolate(p.position(), tetIs) +
	//			p.chargeFlux();
	
	
	const scalar flux = td.chargeDefAverage_().interpolate( p.coordinates(), tetIs ) + p.chargeFlux();  
	    
    	//std::cout<< std::scientific;
    	//std::cout<<"Flux: "<<p.charge()<<" "<<p.chargeFlux()<<" "<<p.chargeDef()<<" "<<
	//			td.chargeDefAverage_().interpolate( p.position(), tetIs )<<
	//			" "<<flux<<std::endl;
    
	pcharge += flux;
	p.chargeDef()   = 0.0;
	p.chargeFlux() = 0.0;
	
    }  

}
*/
template<class CloudType>
template<class TrackCloudType>
void Foam::MPPICHWCloud<CloudType>::motion
(
    TrackCloudType& cloud,
    typename parcelType::trackingData& td
)
{
    
    // Kinematic
    // ~~~~~~~~~
    
    // force calculation and tracking
	td.part() = parcelType::trackingData::tpLinearTrack;    
    
    /*
    //Charge transfer
    setChargeCellValues( td );
    td.updateAverages(*this);
    updateParcelCharge( td );
    */
    
    CloudType::move(cloud, td, this->db().time().deltaTValue());
    
    // Preliminary
    // ~~~~~~~~~~~

    // switch forces off so they are not applied in corrector steps
    this->forces().setCalcNonCoupled(false);
    this->forces().setCalcCoupled(false);


    // Damping
    // ~~~~~~~

    if (dampingModel_->active())
    {
        // update averages
        td.updateAverages(cloud);

        // memory allocation and eulerian calculations
        dampingModel_->cacheFields(true);

        // calculate the damping velocity corrections without moving the parcels
        td.part() = parcelType::trackingData::tpDampingNoTrack;
        CloudType::move(cloud, td, this->db().time().deltaTValue());
	
        // correct the parcel positions and velocities
        td.part() = parcelType::trackingData::tpCorrectTrack;
        CloudType::move(cloud, td, this->db().time().deltaTValue());
	
        // finalise and free memory
        dampingModel_->cacheFields(false);
    }


    // Packing
    // ~~~~~~~

    if (packingModel_->active())
    {
        // same procedure as for damping
        td.updateAverages(cloud);
        packingModel_->cacheFields(true);
	
        td.part() = parcelType::trackingData::tpPackingNoTrack;
        CloudType::move(cloud, td, this->db().time().deltaTValue());
	
        td.part() = parcelType::trackingData::tpCorrectTrack;
        CloudType::move(cloud, td, this->db().time().deltaTValue());
	
        packingModel_->cacheFields(false);
    }


    // Isotropy
    // ~~~~~~~~

    if (isotropyModel_->active())
    {
        // update averages
        td.updateAverages(cloud);

        // apply isotropy model
        isotropyModel_->calculate();
    }
    
    // Final
    // ~~~~~

    // update cell occupancy
    this->updateCellOccupancy();

    // switch forces back on
    this->forces().setCalcNonCoupled(true);
    this->forces().setCalcCoupled(this->solution().coupled());
    
    
}


template<class CloudType>
void Foam::MPPICHWCloud<CloudType>::info()
{
    CloudType::info();

    tmp<volScalarField> alpha = this->theta();

    const scalar alphaMin = gMin(alpha().primitiveField());
    const scalar alphaMax = gMax(alpha().primitiveField());

    Info<< "    Min cell volume fraction        = " << alphaMin << endl;
    Info<< "    Max cell volume fraction        = " << alphaMax << endl;

    if (alphaMax < SMALL)
    {
        return;
    }

    scalar nMin = GREAT;

    forAll(this->mesh().cells(), cellI)
    {
        const label n = this->cellOccupancy()[cellI].size();

        if (n > 0)
        {
            const scalar nPack = n*alphaMax/alpha()[cellI];

            if (nPack < nMin)
            {
                nMin = nPack;
            }
        }
    }

    reduce(nMin, minOp<scalar>());

    Info<< "    Min dense number of parcels     = " << nMin << endl;
}


// ************************************************************************* //
