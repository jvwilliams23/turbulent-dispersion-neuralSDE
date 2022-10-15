/*volVectorField tau =  - fvc::div((rhoc*this->nuEff())*dev2(T(fvc::grad(Uc))))
                      - fvm::laplacian(rhoc*this->nuEff(), Uc);
*/
//volVectorField viscStresses = -fvc::laplacian(muc, Uc);


fvVectorMatrix UcEqn
(
    fvm::ddt(Uc) + fvm::div(phic, Uc)
    //- fvm::laplacian(continuousPhaseTransport.nu(), Uc)
    + continuousPhaseTurbulence->divDevRhoReff(Uc)
 ==
    (0.0/rhoc)*cloudSU
  + fvOptions(Uc) //JW 07/12/2020
);

UcEqn.relax();

fvOptions.constrain(UcEqn);  //JW 07/12/2020

volScalarField rAUc(1.0/UcEqn.A());
surfaceScalarField rAUcf("Dp", fvc::interpolate(rAUc));

surfaceScalarField phicForces
(
   fvc::flux(rAUc*0.0*cloudVolSUSu/rhoc) + rAUcf*(g & mesh.Sf())
);

if (pimple.momentumPredictor())
{
    solve
    (
        UcEqn
     ==
        fvc::reconstruct
        (
            phicForces/rAUcf - fvc::snGrad(p)*mesh.magSf()
        )
    );
    fvOptions.correct(Uc); //JW 07/12/2020
}
