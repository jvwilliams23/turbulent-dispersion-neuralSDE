/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2015-2018 OpenFOAM Foundation
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
    Foam::LESModels::WALEwFields

Description
    The Wall-adapting local eddy-viscosity (WALEwFields) SGS model.

    Reference:
    \verbatim
        Nicoud, F., & Ducros, F. (1999).
        Subgrid-scale stress modelling based on the square of the velocity
        gradient tensor.
        Flow, Turbulence and Combustion, 62(3), 183-200.
    \endverbatim

    The default model coefficients are
    \verbatim
        WALEwFieldsCoeffs
        {
            Ck                  0.094;
            Ce                  1.048;e
            Cw                  0.325;
        }
    \endverbatim

See also
    Foam::LESModels::Smagorinsky

SourceFiles
    WALEwFields.C

\*---------------------------------------------------------------------------*/

#ifndef WALEwFields_H
#define WALEwFields_H

#include "LESModel.H"
#include "LESeddyViscosity.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

/*---------------------------------------------------------------------------*\
                           Class WALEwFields Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class WALEwFields
:
    public LESeddyViscosity<BasicTurbulenceModel>
{
    // Private Member Functions

        // Disallow default bitwise copy construct and assignment
        WALEwFields(const WALEwFields&);
        void operator=(const WALEwFields&);


protected:

    // Protected data
        volScalarField      k_;
        volScalarField      epsilon_;
        dimensionedScalar   Ck_;
        dimensionedScalar   Cw_;


    // Protected Member Functions

        //- Return the deviatoric symmetric part of the square of the given
        //  velocity gradient field
        tmp<volSymmTensorField> Sd(const volTensorField& gradU) const;

        //- Return SGS kinetic energy
        //  calculated from the given velocity gradient
        tmp<volScalarField> k(const volTensorField& gradU) const;

        //- Update the SGS eddy-viscosity
        virtual void correctNut();


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    //- Runtime type information
    TypeName("WALEwFields");


    // Constructors

        //- Construct from components
        WALEwFields
        (
            const alphaField& alpha,
            const rhoField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName = turbulenceModel::propertiesName,
            const word& type = typeName
        );


    //- Destructor
    virtual ~WALEwFields()
    {}


    // Member Functions

        //- Read model coefficients if they have changed
        virtual bool read();

        //- Return SGS kinetic energy
        virtual tmp<volScalarField> k() const
        {
            return k(fvc::grad(this->U_));
        }

        //- Return sub-grid disipation rate
        virtual tmp<volScalarField> epsilon() const;

        //- Correct Eddy-Viscosity and related properties
        virtual void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "WALEwFields.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
