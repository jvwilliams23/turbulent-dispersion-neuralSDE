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

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class CloudType>
inline const Foam::MPPICHWCloud<CloudType>&
Foam::MPPICHWCloud<CloudType>::cloudCopy() const
{
    return cloudCopyPtr_();
}


template<class CloudType>
inline const Foam::PackingModel<Foam::MPPICHWCloud<CloudType> >&
Foam::MPPICHWCloud<CloudType>::packingModel() const
{
    return packingModel_();
}


template<class CloudType>
inline Foam::PackingModel<Foam::MPPICHWCloud<CloudType> >&
Foam::MPPICHWCloud<CloudType>::packingModel()
{
    return packingModel_();
}


template<class CloudType>
inline const Foam::DampingModel<Foam::MPPICHWCloud<CloudType> >&
Foam::MPPICHWCloud<CloudType>::dampingModel() const
{
    return dampingModel_();
}


template<class CloudType>
inline Foam::DampingModel<Foam::MPPICHWCloud<CloudType> >&
Foam::MPPICHWCloud<CloudType>::dampingModel()
{
    return dampingModel_();
}


template<class CloudType>
inline const Foam::IsotropyModel<Foam::MPPICHWCloud<CloudType> >&
Foam::MPPICHWCloud<CloudType>::isotropyModel() const
{
    return isotropyModel_();
}


template<class CloudType>
inline Foam::IsotropyModel<Foam::MPPICHWCloud<CloudType> >&
Foam::MPPICHWCloud<CloudType>::isotropyModel()
{
    return isotropyModel_();
}

/*
template<class CloudType>
inline Foam::volScalarField&
Foam::MPPICHWCloud<CloudType>::charge()
{
    return charge_;
}

template<class CloudType>
inline Foam::volScalarField&
Foam::MPPICHWCloud<CloudType>::chargeDef()
{
    return chargeDef_;
}

template<class CloudType>
inline Foam::volScalarField&
Foam::MPPICHWCloud<CloudType>::workFunction()
{
    return workFunction_;
}

template<class CloudType>
inline Foam::volScalarField&
Foam::MPPICHWCloud<CloudType>::granularTemperature()
{
    return granularTemperature_;
}

template<class CloudType>
inline Foam::volScalarField&
Foam::MPPICHWCloud<CloudType>::volumeFraction()
{
    return volumeFraction_;
}

template<class CloudType>
inline const Foam::volVectorField&
Foam::MPPICHWCloud<CloudType>::electricField() const
{
    return electricField_;
}


template<class CloudType>
inline Foam::volVectorField&
Foam::MPPICHWCloud<CloudType>::us()
{
    return us_;
}

template<class CloudType>
inline const Foam::tmp<Foam::volScalarField>
Foam::MPPICHWCloud<CloudType>::rhoe() const
{
    
    autoPtr<AveragingMethod<scalar>> rhoeAveragePtr
    (
        AveragingMethod<scalar>::New
        (
            IOobject
            (
                this->name() + ":rhoeAverage",
                this->db().time().timeName(),
                this->mesh()
            ),
            this->solution().dict(),
            this->mesh()
        )
    );
    
    AveragingMethod<scalar>& rhoeAverage = rhoeAveragePtr(); 
       
    forAllConstIter(typename CloudType, *this, iter)
    {
        const parcelType& p = iter();
        const tetIndices tetIs = p.currentTetIndices();
        rhoeAverage.add(p.coordinates(), tetIs, p.charge());
    }

    tmp<volScalarField> trhoeEff
    (
        new volScalarField
        (
            IOobject
            (
                this->name() + ":rhoeEff",
                this->db().time().timeName(),
                this->db(),
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            this->mesh(),
            dimensionedScalar("zero", dimDensity, 0.0)
        )
    );
    volScalarField& rhoe = trhoeEff.ref();
    rhoe.primitiveFieldRef() = rhoeAverage.primitiveField();
    
    return trhoeEff;
}

template<class CloudType>
inline void
Foam::MPPICHWCloud<CloudType>::updateRhoe()
{

	rhoe_.primitiveFieldRef() = 0.0;

    forAllConstIter(typename MPPICHWCloud<CloudType>, *this, iter)
    {
        const parcelType& p = iter();

        rhoe_[p.cell()] += p.charge();

    }
    
    rhoe_.primitiveFieldRef() /= this->mesh().V();

}
*/


// ************************************************************************* //
