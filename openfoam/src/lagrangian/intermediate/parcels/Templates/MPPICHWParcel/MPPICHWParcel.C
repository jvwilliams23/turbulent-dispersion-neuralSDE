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

#include "MPPICHWParcel.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ParcelType>
Foam::MPPICHWParcel<ParcelType>::MPPICHWParcel
(
    const MPPICHWParcel<ParcelType>& p
)
:
    ParcelType(p),
    UCorrect_(p.UCorrect_)//,
    /*
    charge_(p.charge()),
    chargeFlux_(0.0),
    workFunction_(p.workFunction()),
    Ef_( p.Ef() ),
    chargeI_(p.chargeI()),
    workFunctionI_(p.workFunctionI()),
    chargeDef_( p.chargeDef() ),
    UI_(p.UI()),
    setValuesCalled( false )
    */
{}


template<class ParcelType>
Foam::MPPICHWParcel<ParcelType>::MPPICHWParcel
(
    const MPPICHWParcel<ParcelType>& p,
    const polyMesh& mesh
)
:
    ParcelType(p, mesh),
    UCorrect_(p.UCorrect_)//,
    /*
    charge_(p.charge()),
    chargeFlux_( 0.0 ),
    workFunction_(p.workFunction()),
    Ef_( p.Ef() ),
    chargeI_(p.chargeI()),
    workFunctionI_(p.workFunctionI()),
    chargeDef_( p.chargeDef() ),
    UI_(p.UI()),
    setValuesCalled( false )
    */ 
{}

/*
template<class ParcelType>
template<class TrackCloudType>
void Foam::MPPICHWParcel<ParcelType>::updateCharge
(
	TrackCloudType& cloud,
    trackingData& td,
    const scalar trackTime
)
{
    
    typename TrackCloudType::parcelType& p =
    static_cast<typename TrackCloudType::parcelType&>(*this);
    
    const label celli = p.cell();
    const point start(p.position()); 
    scalar dt = trackTime;

    //update electric field
    tetIndices tetIs = this->currentTetIndices();
    Ef_ = td.EInterp().interpolate(this->coordinates(), tetIs);
    
    UI_ = td.usInterp().interpolate(this->coordinates(), tetIs);

    //update particle position average properties
    
    chargeI_ = td.qInterp().interpolate(this->coordinates(), tetIs);
    workFunctionI_ = td.workFunctionInterp().interpolate(this->coordinates(), tetIs);
    
    volumeFraction_ = td.volumeAverageInterp().interpolate(this->coordinates(), tetIs);
    
    granularTemperature_ = td.granularTemperatureInterp().interpolate( this->coordinates(), tetIs );
    //std::cout<<"Temperature: "<<granularTemperature_<<std::endl;
    
    setValuesCalled = true;
    
    cloud.functions().postMove(p, celli, dt, start, td.keepParticle);
    
}


template<class ParcelType>
template<class TrackData>
void Foam::MPPICHWParcel<ParcelType>::setCellValues
(
    TrackData& td,
    const scalar dt,
    const label cellI
)
{
        
    //Kinematic cloud setCellValues
    ParcelType::setCellValues( td, dt, cellI );
    
}
*/


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class ParcelType>
template<class TrackCloudType>
bool Foam::MPPICHWParcel<ParcelType>::move
(
	TrackCloudType& cloud,
    trackingData& td,
    const scalar trackTime
)
{
    typename TrackCloudType::parcelType& p =
        static_cast<typename TrackCloudType::parcelType&>(*this);

    switch (td.part())
    {
        case trackingData::tpLinearTrack:
        {
            ParcelType::move(cloud, td, trackTime);

            break;
        }
        case trackingData::tpDampingNoTrack:
        {
            p.UCorrect() =
                cloud.dampingModel().velocityCorrection(p, trackTime);

            td.keepParticle = true;
			td.switchProcessor = false;

            break;
        }
        case trackingData::tpPackingNoTrack:
        {
            p.UCorrect() =
                cloud.packingModel().velocityCorrection(p, trackTime);

            td.keepParticle = true;
			td.switchProcessor = false;

            break;
        }
        case trackingData::tpCorrectTrack:
        {
            vector U = p.U();

            scalar f = p.stepFraction();

            scalar a = p.age();

            p.U() = (1.0 - f)*p.UCorrect();

            ParcelType::move(cloud, td, trackTime);

            p.U() = U + (p.stepFraction() - f)*p.UCorrect();

            p.age() = a;

            break;
        }
    }

    return td.keepParticle;
}


// * * * * * * * * * * * * * * IOStream operators  * * * * * * * * * * * * * //

#include "MPPICHWParcelIO.C"

// ************************************************************************* //
