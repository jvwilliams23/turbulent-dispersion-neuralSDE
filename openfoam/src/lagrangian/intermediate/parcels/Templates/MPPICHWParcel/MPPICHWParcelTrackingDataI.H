/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2013 OpenFOAM Foundation
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

#include "AveragingMethod.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class ParcelType>
template<class TrackCloudType>
inline Foam::MPPICHWParcel<ParcelType>::trackingData::trackingData
(
    const TrackCloudType& cloud,
    trackPart part
)
:
    ParcelType::trackingData(cloud),
    volumeAverage_
    (
        AveragingMethod<scalar>::New
        (
            IOobject
            (
                cloud.name() + ":volumeAverage",
                cloud.db().time().timeName(),
                cloud.mesh()
            ),
            cloud.solution().dict(),
            cloud.mesh()
        )
    ),
    radiusAverage_
    (
        AveragingMethod<scalar>::New
        (
            IOobject
            (
                cloud.name() + ":radiusAverage",
                cloud.db().time().timeName(),
                cloud.mesh()
            ),
            cloud.solution().dict(),
            cloud.mesh()
        )
    ),
    rhoAverage_
    (
        AveragingMethod<scalar>::New
        (
            IOobject
            (
                cloud.name() + ":rhoAverage",
                cloud.db().time().timeName(),
                cloud.mesh()
            ),
            cloud.solution().dict(),
            cloud.mesh()
        )
    ),
    uAverage_
    (
        AveragingMethod<vector>::New
        (
            IOobject
            (
                cloud.name() + ":uAverage",
                cloud.db().time().timeName(),
                cloud.mesh()
            ),
            cloud.solution().dict(),
            cloud.mesh()
        )
    ),
    uSqrAverage_
    (
        AveragingMethod<scalar>::New
        (
            IOobject
            (
                cloud.name() + ":uSqrAverage",
                cloud.db().time().timeName(),
                cloud.mesh()
            ),
            cloud.solution().dict(),
            cloud.mesh()
        )
    ),
    frequencyAverage_
    (
        AveragingMethod<scalar>::New
        (
            IOobject
            (
                cloud.name() + ":frequencyAverage",
                cloud.db().time().timeName(),
                cloud.mesh()
            ),
            cloud.solution().dict(),
            cloud.mesh()
        )
    ),
    massAverage_
    (
        AveragingMethod<scalar>::New
        (
            IOobject
            (
                cloud.name() + ":massAverage",
                cloud.db().time().timeName(),
                cloud.mesh()
            ),
            cloud.solution().dict(),
            cloud.mesh()
        )
    ),
    /*
    qAverage_
    (
        AveragingMethod<scalar>::New
        (
            IOobject
            (
                cloud.name() + ":qAverage",
                cloud.db().time().timeName(),
                cloud.mesh()
            ),
            cloud.solution().dict(),
            cloud.mesh()
        )
    ),
    rhoeAverage_
    (
        AveragingMethod<scalar>::New
        (
            IOobject
            (
                cloud.name() + ":rhoeAverage",
                cloud.db().time().timeName(),
                cloud.mesh()
            ),
            cloud.solution().dict(),
            cloud.mesh()
        )
    ),
    workFunctionAverage_
    (
        AveragingMethod<scalar>::New
        (
            IOobject
            (
                cloud.name() + ":workFunctionAverage",
                cloud.db().time().timeName(),
                cloud.mesh()
            ),
            cloud.solution().dict(),
            cloud.mesh()
        )
    ),
    EInterp_
    (
        interpolation<vector>::New
        (
            cloud.solution().interpolationSchemes(),
            cloud.electricField()
        )
    ),
    qInterp_
    (
        interpolation<scalar>::New
        (
            cloud.solution().interpolationSchemes(),
            cloud.charge()
        )
    ),
    workFunctionInterp_
    (
        interpolation<scalar>::New
        (
            cloud.solution().interpolationSchemes(),
            cloud.workFunction()
        )
    ),
    usInterp_
    (
        interpolation<vector>::New
        (
            cloud.solution().interpolationSchemes(),
            cloud.us()
        )
    ),
    volumeAverageInterp_
    (
        interpolation<scalar>::New
        (
            cloud.solution().interpolationSchemes(),
            cloud.volumeFraction()
        )
    ),
    granularTemperatureInterp_
    (
        interpolation<scalar>::New
        (
            cloud.solution().interpolationSchemes(),
            cloud.granularTemperature()
        )
    ),
    chargeDefInterp_
    (
        interpolation<scalar>::New
        (
            cloud.solution().interpolationSchemes(),
            cloud.chargeDef()
        )
    ),
    chargeDefAverage_
    (
        AveragingMethod<scalar>::New
        (
            IOobject
            (
                cloud.name() + ":chargeDefAverage",
                cloud.db().time().timeName(),
                cloud.mesh()
            ),
            cloud.solution().dict(),
            cloud.mesh()
        )
    ),
    */
    part_(part)
{}

/*
template<class ParcelType>
template<class TrackCloudType>
inline void
Foam::MPPICHWParcel<ParcelType>::trackingData::updateChargeDeficit
(
    const TrackCloudType& cloud
)
{
    
    // temporary weights
    autoPtr<AveragingMethod<scalar> > numberAveragePtr
    (
        AveragingMethod<scalar>::New
        (
            IOobject
            (
                cloud.name() + ":numberAverage",
                cloud.db().time().timeName(),
                cloud.mesh()
            ),
            cloud.solution().dict(),
            cloud.mesh()
        )
    );
    
    AveragingMethod<scalar>& numberAverage = numberAveragePtr();
    
    chargeDefAverage_() = 0.0;
    numberAverage = 0.0;
    
    // averaging sums
    forAllConstIter(typename TrackCloudType, cloud, iter)
    {
        const typename TrackCloudType::parcelType& p = iter();
        const tetIndices tetIs = p.currentTetIndices();
		const scalar m = p.chargeDef() * cloud.mesh().V()[tetIs.cell()];

        chargeDefAverage_->add(p.coordinates(), tetIs, m );
		numberAverage.add(p.coordinates(), tetIs, cloud.mesh().V()[tetIs.cell()] );
    }  
        
    chargeDefAverage_->average(numberAverage);
    
    volScalarField& chargeDef = cloud.chargeDef();
    
    //update charge deficit to mesh
    chargeDef.primitiveFieldRef() = chargeDefAverage_->primitiveField();
    
    //volScalarField& chargeDef = cloud.chargeDef();
    
    //chargeDef = 0.0;
    
    //forAllConstIter(typename CloudType, cloud, iter)
    //{
    //    const typename CloudType::parcelType& p = iter();
    //    const label celli = p.cell();
    //	chargeDef[celli] += p.chargeDef();
    //}

}
*/



template<class ParcelType>
template<class TrackCloudType>
inline void Foam::MPPICHWParcel<ParcelType>::trackingData::updateAverages
(
    const TrackCloudType& cloud
)
{
    // zero the sums
    volumeAverage_() = 0;
    radiusAverage_() = 0;
    rhoAverage_() = 0;
    uAverage_() = vector::zero;
    uSqrAverage_() = 0;
    frequencyAverage_() = 0;
    massAverage_() = 0;

    // temporary weights
    autoPtr<AveragingMethod<scalar> > weightAveragePtr
    (
        AveragingMethod<scalar>::New
        (
            IOobject
            (
                cloud.name() + ":weightAverage",
                cloud.db().time().timeName(),
                cloud.mesh()
            ),
            cloud.solution().dict(),
            cloud.mesh()
        )
    );
    AveragingMethod<scalar>& weightAverage = weightAveragePtr();

    /*
    // temporary weights
    autoPtr<AveragingMethod<scalar> > numberAveragePtr
    (
        AveragingMethod<scalar>::New
        (
            IOobject
            (
                cloud.name() + ":numberAverage",
                cloud.db().time().timeName(),
                cloud.mesh()
            ),
            cloud.solution().dict(),
            cloud.mesh()
        )
    );
    
    
    AveragingMethod<scalar>& numberAverage = numberAveragePtr();
    */

    // averaging sums
    forAllConstIter(typename TrackCloudType, cloud, iter)
    {
        const typename TrackCloudType::parcelType& p = iter();
        const tetIndices tetIs = p.currentTetIndices();

        const scalar m = p.nParticle()*p.mass();

        volumeAverage_->add(p.coordinates(), tetIs, p.nParticle()*p.volume());
        rhoAverage_->add(p.coordinates(), tetIs, m*p.rho());
        uAverage_->add(p.coordinates(), tetIs, m*p.U());
        massAverage_->add(p.coordinates(), tetIs, m);
    }
    
    volumeAverage_->average();
    massAverage_->average();
    rhoAverage_->average(massAverage_);
    uAverage_->average(massAverage_);

    // squared velocity deviation
    forAllConstIter(typename TrackCloudType, cloud, iter)
    {
        const typename TrackCloudType::parcelType& p = iter();
        const tetIndices tetIs = p.currentTetIndices();

        const vector u = uAverage_->interpolate(p.coordinates(), tetIs);

        uSqrAverage_->add
        (
            p.coordinates(),
            tetIs,
            p.nParticle()*p.mass()*magSqr(p.U() - u)
        );
    }
    uSqrAverage_->average(massAverage_);

    // sauter mean radius
    radiusAverage_() = volumeAverage_();
    weightAverage = 0;
    forAllConstIter(typename TrackCloudType, cloud, iter)
    {
        const typename TrackCloudType::parcelType& p = iter();
        const tetIndices tetIs = p.currentTetIndices();

        weightAverage.add
        (
            p.coordinates(),
            tetIs,
            p.nParticle()*pow(p.volume(), 2.0/3.0)
        );
    }
    weightAverage.average();
    radiusAverage_->average(weightAverage);

    // collision frequency
    weightAverage = 0;
    forAllConstIter(typename TrackCloudType, cloud, iter)
    {
        const typename TrackCloudType::parcelType& p = iter();
        const tetIndices tetIs = p.currentTetIndices();

        const scalar a = volumeAverage_->interpolate(p.coordinates(), tetIs);
        const scalar r = radiusAverage_->interpolate(p.coordinates(), tetIs);
        const vector u = uAverage_->interpolate(p.coordinates(), tetIs);

        const scalar f = 0.75*a/pow3(r)*sqr(0.5*p.d() + r)*mag(p.U() - u);

        frequencyAverage_->add(p.coordinates(), tetIs, p.nParticle()*f*f);

        weightAverage.add(p.coordinates(), tetIs, p.nParticle()*f);
    }
    frequencyAverage_->average(weightAverage);
    
    /*
    //compute mean charge ********************************* 
    qAverage_() = 0.0;
    workFunctionAverage_() = 0.0;
    numberAverage = 0.0;
    
    volScalarField& qField = cloud.charge();
    qField = 0.0;
    
    forAllConstIter(typename TrackCloudType, cloud, iter)
    {
        
	const typename TrackCloudType::parcelType& p = iter();
	const tetIndices tetIs = p.currentTetIndices();
	
	const scalar vol = p.nParticle() * cloud.mesh().V()[tetIs.cell()];
	
	qAverage_->add( p.coordinates(), tetIs, vol*p.charge() );
	numberAverage.add( p.coordinates(), tetIs, vol );	
	
    }
    
    numberAverage.average();
    
    qAverage_->average(numberAverage);
    qField.primitiveFieldRef() = qAverage_->primitiveField();
        
    //update volume fraction
    volScalarField& volumeFraction = cloud.volumeFraction();
    volumeFraction.primitiveFieldRef() = volumeAverage_->primitiveField();
    
    
    //compute mean work function
    volScalarField& workFunctionField = cloud.workFunction();
    workFunctionField = 0.0;
    
    forAllConstIter(typename TrackCloudType, cloud, iter)
    {
        
	const typename TrackCloudType::parcelType& p = iter();
	const tetIndices tetIs = p.currentTetIndices();
	
	const scalar m = p.nParticle() * cloud.mesh().V()[tetIs.cell()];
	
	workFunctionAverage_->add(p.coordinates(), tetIs, m*p.workFunction() );
	
    }
    
    
    
    //update work function values at mesh
    workFunctionAverage_->average(numberAverage);
    workFunctionField.primitiveFieldRef() = workFunctionAverage_->primitiveField();
   
    //update granular temperature at mesh
    volScalarField& granularTemperatureField = cloud.granularTemperature();
    granularTemperatureField.primitiveFieldRef() = uSqrAverage_->primitiveField();
    */      
}

template<class ParcelType>
inline typename Foam::MPPICHWParcel<ParcelType>::trackingData::trackPart
Foam::MPPICHWParcel<ParcelType>::trackingData::part() const
{
    return part_;
}


template<class ParcelType>
inline typename Foam::MPPICHWParcel<ParcelType>::trackingData::trackPart&
Foam::MPPICHWParcel<ParcelType>::trackingData::part()
{
    return part_;
}

/*
template<class ParcelType>
inline const Foam::interpolation<Foam::vector>&
Foam::MPPICHWParcel<ParcelType>::trackingData::EInterp() const
{
    return EInterp_();
}


template<class ParcelType>
inline const Foam::interpolation<Foam::vector>&
Foam::MPPICHWParcel<ParcelType>::trackingData::usInterp() const
{
    return usInterp_();
}

template<class ParcelType>
inline const Foam::interpolation<Foam::scalar>&
Foam::MPPICHWParcel<ParcelType>::trackingData::qInterp() const
{
    return qInterp_();
}

template<class ParcelType>
inline const Foam::interpolation<Foam::scalar>&
Foam::MPPICHWParcel<ParcelType>::trackingData::workFunctionInterp() const
{
    return workFunctionInterp_();
}

template<class ParcelType>
inline const Foam::interpolation<Foam::scalar>&
Foam::MPPICHWParcel<ParcelType>::trackingData::volumeAverageInterp() const
{
    return volumeAverageInterp_();
}


template<class ParcelType>
inline const Foam::interpolation<Foam::scalar>&
Foam::MPPICHWParcel<ParcelType>::trackingData::granularTemperatureInterp() const
{
    return granularTemperatureInterp_();
}


template<class ParcelType>
inline const Foam::interpolation<Foam::scalar>&
Foam::MPPICHWParcel<ParcelType>::trackingData::chargeDefInterp() const
{
    return chargeDefInterp_();
}
*/
// ************************************************************************* //
