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

Application
    postParticleChannel

Description
    Post-processes data from channel flow calculations.

    For each time: calculate: txx, txy,tyy, txy,
    eps, prod, vorticity, enstrophy and helicity. Assuming that the mesh
    is periodic in the x and z directions, collapse Umeanx, Umeany, txx,
    txy and tyy to a line and print them as standard output.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "channelIndex.H"
#include "makeGraph.H"

#include "OSspecific.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::noParallel();
    timeSelector::addOptions();

    argList::addBoolOption
    (
        "concentration",
        "write instantaneous and averaged concentration (C/Cin)"
    );

    argList::addBoolOption
    (
        "mean",
        "write mean fields"
    );

    #include "setRootCase.H"
    #include "createTime.H"

    // Get times list
    instantList timeDirs = timeSelector::select0(runTime, args);

    #include "createMesh.H"
    #include "readTransportProperties.H"

    const word& gFormat = runTime.graphFormat();

    // Setup channel indexing for averaging over channel down to a line

    IOdictionary channelDict
    (
        IOobject
        (
            "postParticleChannelDict",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    );
    const word UName = channelDict.lookup("UName");
    const word alphaName = channelDict.lookup("alphaName");


    channelIndex channelIndexing(mesh, channelDict);


    if ( args.optionFound("mean") )
    {
        // For each time step read all fields
        forAll(timeDirs, timeI)
        {
            runTime.setTime(timeDirs[timeI], timeI);
            Info<< "Collapsing fields for time " << runTime.timeName() << endl;

            #include "readFields.H"
            #include "calculateFields.H"

            // Average fields over channel down to a line
            #include "collapse.H"
        }
    }
    if ( args.optionFound("concentration") )
    {
        bool c_inFound = false;
        volScalarField alphaInit
        (
            IOobject
            (
                "alpha_init",
                runTime.timeName(),
                mesh,
                IOobject::READ_IF_PRESENT
            ),
            mesh,
            dimensionedScalar("0", dimless, Zero)
        );

        forAll(timeDirs, timeI)
        {
            runTime.setTime(timeDirs[timeI], timeI);
            Info<< "Collapsing concentration fields for time " << runTime.timeName() << endl;
            IOobject alphaHeader
            (
                alphaName,
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ
            );
            IOobject alphaMeanHeader
            (   
                alphaName+"Mean",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ
            );
            if (!alphaHeader.typeHeaderOk<volScalarField>(true)) 
            {
                Info << "No alpha field found" << endl;
                continue;
            }
            volScalarField alphaInstant
            (
                alphaHeader,
                mesh
            );

            if ( !c_inFound )
            {
                c_inFound = true;
                alphaInit = alphaInstant;
                Info << tab << "Found initial alpha field at time = " << timeI << endl;
            }
            const scalarField& y = channelIndexing.y();

            //volScalarField conc = (1.0-alphaInstant)/max(1.0-alphaInit,VSMALL);
            volScalarField alphapInstant = 1.0 - alphaInstant;
            volScalarField alphapInit = 1.0 - alphaInit;


            fileName path(alphaInstant.rootPath()/alphaInstant.caseName()/"graphs"/alphaInstant.instance());
            mkDir(path);
            Info << tab << "Writing to" << path << endl;
            //- get filtered particle velocity components
            /*scalarField concValues
            (
                channelIndexing.collapse(conc)
            );*/
            scalarField alphapInstantValues
            (
                channelIndexing.collapse(alphapInstant)
            );
            scalarField alphapInitValues
            (
                channelIndexing.collapse(alphapInit)
            );
            scalarField concValues = alphapInstantValues/max(alphapInitValues, VSMALL);

            makeGraph(y, concValues, "concInstant", path, gFormat);
            makeGraph(y, alphapInstantValues, "alphapInstant", path, gFormat);            

            if ( alphaMeanHeader.typeHeaderOk<volScalarField>(true))
            {
                volScalarField alphaMean
                (
                    alphaMeanHeader,
                    mesh
                );
                volScalarField alphapMean = 1.0 - alphaMean;
                scalarField alphapMeanValues
                (
                    channelIndexing.collapse(alphapMean)
                );
                scalarField concMeanValues = alphapMeanValues / max(alphapInitValues, VSMALL);
                makeGraph(y, concMeanValues, "concMean", path, gFormat);    
            }

        }
    }


    Info<< "\nEnd\n" << endl;

    return 0;
}


// ************************************************************************* //
