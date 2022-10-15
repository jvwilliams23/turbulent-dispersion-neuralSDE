/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2013-2015 OpenFOAM Foundation
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
#include "IOstreams.H"
#include "IOField.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

template<class ParcelType>
Foam::string Foam::MPPICHWParcel<ParcelType>::propertyList_ =
    Foam::MPPICHWParcel<ParcelType>::propertyList();

template<class ParcelType>
const std::size_t Foam::MPPICHWParcel<ParcelType>::sizeofFields_
(
    sizeof(MPPICHWParcel<ParcelType>) - sizeof(ParcelType)
);


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ParcelType>
Foam::MPPICHWParcel<ParcelType>::MPPICHWParcel
(
    const polyMesh& mesh,
    Istream& is,
    bool readFields
)
:
    ParcelType(mesh, is, readFields),
    UCorrect_(vector::zero)
{
    if (readFields)
    {
        if (is.format() == IOstream::ASCII)
        {
            is >> UCorrect_;
	    /*
			is >> charge_;
			is >> chargeFlux_;
			is >> workFunction_;
			is >> Ef_;
	   */		
        }
        else
        {
            is.read(reinterpret_cast<char*>(&UCorrect_), sizeofFields_);
        }
    }

    is.check
    (
        "MPPICHWParcel<ParcelType>::Collisions"
        "(const polyMesh&, Istream&, bool)"
    );
}


template<class ParcelType>
template<class CloudType>
void Foam::MPPICHWParcel<ParcelType>::readFields(CloudType& c)
{
    if (!c.size())
    {
        return;
    }

    ParcelType::readFields(c);

//	scalar defaultWorkFunction = c.constProps().dict().lookupOrDefault("defaultWorkFunction", 0.0);
//	scalar defaultCharge = c.constProps().dict().lookupOrDefault("defaultCharge", 0.0);

    IOField<vector> UCorrect(c.fieldIOobject("UCorrect", IOobject::MUST_READ));
    c.checkFieldIOobject(c, UCorrect);
/*
    IOField<scalar> charge
	(
		c.fieldIOobject("charge", IOobject::READ_IF_PRESENT), 
		Field<scalar>(c.size(), defaultCharge)
	);

    IOField<scalar> workFunction
	(
		c.fieldIOobject("workFunction", IOobject::READ_IF_PRESENT), 
		Field<scalar>(c.size(), defaultWorkFunction)
	);
*/	
    label i = 0;

    forAllIter(typename CloudType, c, iter)
    {
        MPPICHWParcel<ParcelType>& p = iter();

        p.UCorrect_ = UCorrect[i];
	/*
		p.workFunction_ = workFunction[i];
		p.charge_ = charge[i];
		p.chargeFlux_ = 0.0;
		p.Ef_ = Zero;
		p.chargeI_  = 0.0;
		p.workFunctionI_  = 0.0;
		p.chargeDef_  = 0.0;
		p.UI_  = Zero;
		p.granularTemperature_  = 0.0;
		p.volumeFraction_  = 0.0;
	*/
        i++;
    }
}


template<class ParcelType>
template<class CloudType>
void Foam::MPPICHWParcel<ParcelType>::writeFields(const CloudType& c)
{
    ParcelType::writeFields(c);

    label np =  c.size();

    IOField<vector>
        UCorrect(c.fieldIOobject("UCorrect", IOobject::NO_READ), np);

    /*
    IOField<scalar>
        charge(c.fieldIOobject("charge", IOobject::NO_READ), np);
    */
    
    label i = 0;

    forAllConstIter(typename CloudType, c, iter)
    {
        const MPPICHWParcel<ParcelType>& p = iter();

        UCorrect[i] = p.UCorrect();
	//charge[i] = p.charge();
        i++;
    }

    UCorrect.write();
    //charge.write();
}


// * * * * * * * * * * * * * * * IOstream Operators  * * * * * * * * * * * * //

template<class ParcelType>
Foam::Ostream& Foam::operator<<
(
    Ostream& os,
    const MPPICHWParcel<ParcelType>& p
)
{
    if (os.format() == IOstream::ASCII)
    {
        os  << static_cast<const ParcelType&>(p)
            << token::SPACE << p.UCorrect();
/*
          	<< token::SPACE << p.charge_
            << token::SPACE << p.chargeFlux_
            << token::SPACE << p.workFunction_
            << token::SPACE << p.Ef_;
*/
    }
    else
    {
        os  << static_cast<const ParcelType&>(p);
        os.write
        (
            reinterpret_cast<const char*>(&p.UCorrect_),
            MPPICHWParcel<ParcelType>::sizeofFields_
        );
    }

    os.check
    (
        "Ostream& operator<<(Ostream&, const MPPICHWParcel<ParcelType>&)"
    );

    return os;
}


// ************************************************************************* //
