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

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ParcelType>
inline Foam::MPPICHWParcel<ParcelType>::MPPICHWParcel
(
    const polyMesh& owner,
    const vector& position,
    const label cellI,
    const label tetFaceI,
    const label tetPtI
)
:
    ParcelType(owner, position, cellI, tetFaceI, tetPtI),
    UCorrect_(vector::zero)//,
/*
    charge_(0.0),
    workFunction_(0.0),
    chargeI_(0.0),
    workFunctionI_( 0.0 ),
    chargeDef_(0.0 ),
    UI_(vector::zero),
    chargeFlux_( 0.0 )
*/
{}

template<class ParcelType>
inline Foam::MPPICHWParcel<ParcelType>::MPPICHWParcel
(
    const polyMesh& owner,
    const vector& position,
    const label celli
)
:
    ParcelType(owner, position, celli),
    UCorrect_(vector::zero)//,
    /*
    charge_(0.0),
    workFunction_(0.0),
    chargeI_(0.0),
    workFunctionI_( 0.0 ),
    chargeDef_(0.0 ),
    UI_(vector::zero),
    chargeFlux_( 0.0 )
    */
{}


template<class ParcelType>
inline Foam::MPPICHWParcel<ParcelType>::MPPICHWParcel
(
    const polyMesh& owner,
    const vector& position,
    const label cellI,
    const label tetFaceI,
    const label tetPtI,
    const label typeId,
    const scalar nParticle0,
    const scalar d0,
    const scalar dTarget0,
    const vector& U0,
    const vector& UCorrect0,
    const typename ParcelType::constantProperties& constProps
)
:
    ParcelType
    (
        owner,
        position,
        cellI,
        tetFaceI,
        tetPtI,
        typeId,
        nParticle0,
        d0,
        dTarget0,
        U0,
        constProps
    ),
    UCorrect_(UCorrect0)//,
    /*
    charge_(0.0),
    workFunction_(0.0),
    chargeFlux_( 0.0 ),
    Ef_(Zero),
    chargeI_(0.0),
    workFunctionI_( 0.0 ),
    chargeDef_( 0.0 ),
    UI_(vector::zero)
    */
{}
/*
template<class ParcelType>
inline Foam::MPPICHWParcel<ParcelType>::MPPICHWParcel
(
    const polyMesh& owner,
    const vector& position,
    const label cellI,
    const label tetFaceI,
    const label tetPtI,
    const label typeId,
    const scalar nParticle0,
    const scalar d0,
    const scalar dTarget0,
    const vector& U0,
    const vector& UCorrect0,
    const typename ParcelType::constantProperties& constProps,
    const scalar& charge0,
    const scalar& workFunction0
)
:
    ParcelType
    (
        owner,
        position,
        cellI,
        tetFaceI,
        tetPtI,
        typeId,
        nParticle0,
        d0,
        dTarget0,
        U0,
        constProps
    ),
    UCorrect_(UCorrect0)//,
    charge_(charge0),
    chargeFlux_( 0.0 ),
    workFunction_(workFunction0),
    Ef_(Zero),
    chargeI_(0.0),
    workFunctionI_( 0.0 ),
    chargeDef_( 0.0 ),
    UI_(vector::zero)
{}
*/

// * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * * //

template<class ParcelType>
inline const Foam::vector& Foam::MPPICHWParcel<ParcelType>::UCorrect() const
{
    return UCorrect_;
}


template<class ParcelType>
inline Foam::vector& Foam::MPPICHWParcel<ParcelType>::UCorrect()
{
    return UCorrect_;
}

/*
template<class ParcelType>
inline const Foam::scalar& Foam::MPPICHWParcel<ParcelType>::charge() const
{
    return charge_;
}


template<class ParcelType>
inline Foam::scalar& Foam::MPPICHWParcel<ParcelType>::charge()
{
    return charge_;
}

template<class ParcelType>
inline const Foam::scalar& Foam::MPPICHWParcel<ParcelType>::chargeI() const
{
    return chargeI_;
}


template<class ParcelType>
inline Foam::scalar& Foam::MPPICHWParcel<ParcelType>::chargeI()
{
    return chargeI_;
}


template<class ParcelType>
inline const Foam::scalar& Foam::MPPICHWParcel<ParcelType>::workFunction() const
{
    return workFunction_;
}


template<class ParcelType>
inline Foam::scalar& Foam::MPPICHWParcel<ParcelType>::workFunction()
{
    return workFunction_;
}

template<class ParcelType>
inline const Foam::vector& Foam::MPPICHWParcel<ParcelType>::Ef() const
{
    return Ef_;
}


template<class ParcelType>
inline Foam::vector& Foam::MPPICHWParcel<ParcelType>::Ef()
{
    return Ef_;
}

template<class ParcelType>
inline const Foam::vector& Foam::MPPICHWParcel<ParcelType>::UI() const
{
    return UI_;
}


template<class ParcelType>
inline Foam::vector& Foam::MPPICHWParcel<ParcelType>::UI()
{
    return UI_;
}


template<class ParcelType>
inline const Foam::scalar& Foam::MPPICHWParcel<ParcelType>::workFunctionI() const
{
    return workFunctionI_;
}


template<class ParcelType>
inline Foam::scalar& Foam::MPPICHWParcel<ParcelType>::workFunctionI()
{
    return workFunctionI_;
}

template<class ParcelType>
inline const Foam::scalar& Foam::MPPICHWParcel<ParcelType>::chargeFlux() const
{
    return chargeFlux_;
}


template<class ParcelType>
inline Foam::scalar& Foam::MPPICHWParcel<ParcelType>::chargeFlux()
{
    return chargeFlux_;
}


template<class ParcelType>
inline const Foam::scalar& Foam::MPPICHWParcel<ParcelType>::chargeDef() const
{
    return chargeDef_;
}


template<class ParcelType>
inline Foam::scalar& Foam::MPPICHWParcel<ParcelType>::chargeDef()
{
    return chargeDef_;
}


template<class ParcelType>
inline const Foam::scalar& Foam::MPPICHWParcel<ParcelType>::volumeFraction() const
{
    return volumeFraction_;
}


template<class ParcelType>
inline Foam::scalar& Foam::MPPICHWParcel<ParcelType>::volumeFraction()
{
    return volumeFraction_;
}


template<class ParcelType>
inline const Foam::scalar& Foam::MPPICHWParcel<ParcelType>::granularTemperature() const
{
    return granularTemperature_;
}


template<class ParcelType>
inline Foam::scalar& Foam::MPPICHWParcel<ParcelType>::granularTemperature()
{
    return granularTemperature_;
}
*/


// ************************************************************************* //
