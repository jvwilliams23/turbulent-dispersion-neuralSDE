
Info << "Solving SDE" << endl;
if ( debugEuler_ )
{

    Info << tab << "max UpTilde " << tab << max(UpTilde_.oldTime() ).value() << tab << min(UpTilde_.oldTime()).value() << endl;
    Info << tab << "max UsTilde " << tab << max(UsTilde_.oldTime() ).value() << tab << min(UsTilde_.oldTime()).value() << endl;
    Info << tab << "max UcTilde " << tab << max(Uc).value() << tab << min(Uc).value() << endl;
    Info << tab << "max Us-Uc " << tab << max(UsTilde_.oldTime()-Uc.oldTime()).value() << endl; 
    Info << tab << "max Us-Up " << tab << max(UsTilde_.oldTime()-UpTilde_.oldTime()).value() << endl;
    Info << tab << "max Up-Uc " << tab << max(UpTilde_.oldTime()-Uc.oldTime()).value() << endl; 
    Info << tab << "min k " << tab << min(k.oldTime()).value() << tab << "max k " << tab << max(k.oldTime()).value() << endl;
    Info << tab << "min eps " << tab << min(epsilon.oldTime()).value() << tab << "max eps " << tab << max(epsilon.oldTime()).value() << endl;


    if (taupVSMALL_)
    {
        Info << "taupVSMALL is true" << endl;
    }
}
if (debugEuler_) Info << "Resetting pointers" << endl;

keras::DataChunkFlat DFdataChunk( DFNFeatures_, 0.0 );
std::vector<float>& DFfeaturesDrift = DFdataChunk.get_1d_rw();

keras::DataChunkFlat DFdataChunkDiffusion( DFNFeatures_, 0.0 );
std::vector<float>& DFfeaturesDiffusion = DFdataChunkDiffusion.get_1d_rw();

List<DynamicList<typename CloudType::parcelType*>>& cellOccupancy =
this->owner().cellOccupancy();

bool checkForExtrapolation = false;

Info << "Solving SDE for inertial particles " << endl;
//- Solve SDE for Ufp
forAllIter(typename CloudType, this->owner(), iter)
{
    typename CloudType::parcelType& p = iter();
    const tetIndices tetIs = p.currentTetIndices();
    const label celli = p.cell();

    const vector ones = vector(1.,1.,1.);

    p.stochastic() = true; // JW 01/04/21

    if (p.rebound() == true)
    {
        p.UTurbN() = Zero;   
    }

    unsigned int index = 0;
    DFfeaturesDrift.clear();
    DFfeaturesDiffusion.clear();

    p.rebound() = false;

    // call properties from t=n, x=x_p^n
    vector UcTildeAtP = p.UcTildeN();
    vector UpTildeAtP = p.UpTildeN();
    vector UsTildeAtP = p.UsTildeN();
    vector pGradTildeAtP = p.PGradTildeN();
    scalar k_p = mag(p.kN());//+SMALL;
    scalar epsilon_p = mag(p.epsilonN());//+SMALL;

    if (debugParticles_)
    {
        Info << "interpolated properties (w SMALL) are " << endl;
        Info << "UcTildeAtP " << UcTildeAtP << endl;
        Info << "UpTildeAtP " << UpTildeAtP << endl;
        Info << "UsTildeAtP " << UsTildeAtP << endl;    
        Info << "pGradTildeAtP " << pGradTildeAtP << endl;
        Info << "k_p " << k_p << endl;
        Info << "epsilon_p " << epsilon_p << endl;
        Info << "delta " << delta[celli]<< endl;//cubeRootVol << endl;
    }

    //- turbulent time scale for fluid

    if (debugParticles_) Info << "Getting timescales" << endl;
    const scalar tau_kolmogorov = sqrt(nu[celli] / epsilon_p); // timescale
    const scalar eta_kolmogorov = pow((pow(nu[celli],3.0) / epsilon_p), 0.25); //length scale
    const scalar u_kolmogorov = pow(nu[celli] * epsilon_p, 0.25); // velocity scale
    const scalar timescale = k_p / epsilon_p;

    // get NN inputs as normalised variables
    const scalar timescale_norm = timescale / (tau_kolmogorov * timescaleInpMultiplier_);

    vector viscousTerm = nu[celli] * p.UcLaplacianN(); 

    if (not pGradTermOn_)
    {
        pGradTildeAtP = Zero;
    }
    if (not viscousTermOn_)
    {
        viscousTerm = Zero;
    }

    scalar tauP = taup(p.rho(), rho[celli], p.d(),
                        p.UpTildeN(), p.UsTildeN(), 
                        mu[celli]);
    // get drag response time for NN input
    scalar tauPStokes = taup(p.rho(), rho[celli], p.d(),
                        ones, ones,
                        mu[celli]);

    p.tTurb() = tauP; // save particle time scale at t=n  for corrector loop

    vector relVelNorm = p.UpTildeN() - p.UcTildeN();
    if (relVelIsUpMinusUs_)
    {
        relVelNorm = p.UpTildeN() - p.UsTildeN();
    }

    scalar relVelNormMag = mag(relVelNorm);

    if (taupVSMALL_)
    {
        tauP = ROOTVSMALL;
        tauPStokes = ROOTVSMALL;

        relVelNorm = vector(0.0,0.0,0.0);
        relVelNormMag = 0.0;
    }

    scalar csanadyB = 1.0;

    if ((cellOccupancy[p.cell()].size()>=3) and (csanadyFactorsOn_))
    {
        csanadyB = sqrt(1.0 
                    + sqr(0.8) * 3.0 * sqr(relVelNormMag) / (2.0*k_p+SMALL));
    }

    const scalar mesh_size_norm = (delta[celli] / eta_kolmogorov) / meshSizeMultiplier_;

    if (timescale_norm > 1.0 or mesh_size_norm > 1.0)
    {
        checkForExtrapolation = true;
    }

    // 1/timescale (1/T_L = eps/k)
    DFfeaturesDrift.push_back( timescale_norm );
    DFfeaturesDiffusion.push_back( timescale_norm );
    ++index;
    DFfeaturesDrift.push_back( mesh_size_norm );
    DFfeaturesDiffusion.push_back( mesh_size_norm );
    ++index;

    if (not (index == DFNFeatures_))
    {
        FatalErrorInFunction 
        << "number of inputs to NN not same as DFNFeatures" << tab 
        << index << " not equal " << DFNFeatures_ 
        << exit(FatalError);
    }


    if (debugParticles_)
    {
        printf("inputs DFfeaturesDiffusion %f %f \n", DFfeaturesDiffusion[0], DFfeaturesDiffusion[1]);
        Info << "tau_k is " << tau_kolmogorov << endl;
        Info << "eta_k is " << eta_kolmogorov << endl;
        Info << "u_k is " << u_kolmogorov << endl;
        Info << "k / eps is " << k_p / epsilon_p << endl;

        Info << "Uc is " << p.UcTildeN() << endl;
        Info << "Us is " << p.UTurbN() << endl;
        Info << "Up is " << p.UN() << endl;

        Info << "delta T is " << deltaT << endl;
        Info << "timescale norm is " << timescale_norm << endl;
        Info << "mesh size norm is " << mesh_size_norm << endl;
        Info << "relVelNormMag is " << relVelNormMag << endl;
        Info << "tauP is " << tauP << endl;
        Info << "tauPStokes is " << tauPStokes << endl;
        Info << "compute output" << endl;
    }

    //evaluate the neural network model for the current point
    std::vector<float> DFresponseDrift_dt = DFnnDriftModel_.compute_output( &DFdataChunk ); 
    std::vector<float> DFresponseDiffusion = DFnnDiffusionModel_.compute_output( &DFdataChunkDiffusion );
    scalar Goutput = DFresponseDrift_dt.at(0);
    scalar Boutput = DFresponseDiffusion.at(0);

    if (debugParticles_)
    {
        Info << "GMultiplier is " << GMultiplier_ << endl;
        Info << "BMultiplier is " << BMultiplier_ << endl;

        Info << "GNN raw output is " << Goutput << endl;
        Info << "BNN raw output is " << Boutput << endl;
        //Info << "T is " << T << endl;
        Info << "GNormIndex" << GNormIndex_ << endl;
        Info << "BNormIndex" << BNormIndex_ << endl;
    }

    scalar G = (1.0/Goutput) * GMultiplier_ * sqrt(nu[celli]) / pow(timescale_norm, GNormIndex_);

    if (csanadyFactorsOn_)
    {
        G = G * csanadyB;
    }

    scalar B = Boutput * pow(mesh_size_norm, BNormIndex_) * BMultiplier_ * nu[celli] * csanadyB;

    if (G == 0.0 or B == 0.0)
    {
        G = GREAT;
        B = SMALL;
    }
    else
    {
        if (debugParticles_) Info << "B non-zero " << B << " G non-zero " << G << endl; 
    }

    const scalar T = 1.0 / G;

    // convert NN output to our scheme
    const vector Ti = vector(T, T, T);
    
    const vector BiHatSqr = cmptSqr(vector(B, B, B));

    if (debugParticles_)
    {
        Info << "[after multiplier] G is " << G << endl;
        Info << "[after multiplier] B is " << B << endl;
        Info << "[after multiplier] T is " << 1./G << endl;
    }

    vector relativeMotion = ( UpTildeAtP - UcTildeAtP )
                                     & p.UcGradN();
    if (not relMotionTermOn_)
    {
        relativeMotion = Zero;
    }

    const vector CiMinusUc = -pGradTildeAtP + viscousTerm + relativeMotion;// + g[celli];
    const vector CiTi = cmptMultiply(CiMinusUc, Ti) + p.UcTildeN();

    //-Get components of stochastic integral
    vector gammaN = Zero;
    vector OmegaN = Zero;

    if (debugParticles_) Info << "getting theta" << endl;
    vector theta = ones;
    // to avoid singularity when particle timescale and fluid timescale are same
    if (mag(T-tauP) > SMALL)
    {
        theta = T/(T-tauP)*ones;
    }

    scalar E = 1.0 - exp(-deltaT/tauP);
    if (debugParticles_) Info << " Getting D" << endl;
    vector D =  theta*( exp(-deltaT * G )
                - exp(-deltaT/tauP) );
    vector EminD = (E*ones) - D;

    if (debugParticles_) Info << "Getting gamma etc..." << endl;
    vector gamSqr = gammaLowerSqr(deltaT, BiHatSqr, theta, Ti, tauP);
    if (debugParticles_) Info << "GamSqr" << endl;
    vector GamSqr = GammaUpperSqr(deltaT, BiHatSqr, theta, Ti, tauP);
    if (debugParticles_) Info << "GamOme" << endl;
    vector GamOme = GammaOmega(deltaT, BiHatSqr, theta, Ti, tauP);
    vector gamGam = gammaGamma(deltaT, BiHatSqr, theta, Ti, tauP);
    vector gamOme = gammaOmega(deltaT, BiHatSqr, theta, Ti, tauP);
    vector omeSqr = OmegaSqr(deltaT, BiHatSqr, theta, Ti, tauP);

    omeSqr = cmptMultiply(cmptMultiply(omeSqr, BiHatSqr), cmptSqr(theta));
    GamOme = cmptMultiply(cmptMultiply(GamOme, BiHatSqr), cmptSqr(theta));

    if (debugParticles_ )//&& mag(p.U()) > 10 )
    {
        Info << "deltaT " << deltaT << endl; 
        Info << "BiHatSqr " << BiHatSqr << endl;
        Info << "theta " << theta << endl;
        Info << "Ti " << Ti << endl;
        Info << "tauP " << tauP << endl;
        Info << "gamSqr" << gamSqr << endl;
        Info << "GamSqr" << GamSqr << endl;
        Info << "omeSqr" << omeSqr << endl;
        Info << "gamGam" << gamGam << endl;
        Info << "gamOme" << gamOme << endl;
        Info << "GamOme" << GamOme << endl;
    }

    if (debugParticles_) Info << "Getting cov matrix" << endl;
    vector P11 = cmptSqrt(gamSqr);
    vector P21 = cmptDivideHW(gamOme, cmptSqrt(gamSqr) );
    vector P22 = cmptSqrt(omeSqr - cmptDivideHW( cmptSqr(gamOme), gamSqr ));
    vector P31 = cmptDivideHW(gamGam, cmptSqrt(gamSqr));  // <- maybe the issue
    vector P32 = cmptDivideHW(GamOme-cmptMultiply(P21,P31),P22);
    if (debugParticles_ )//&& mag(p.U()) > 10 )
    {
        Info << "P11" << P11 << endl;
        Info << "P31" << P31 << endl;
        Info << "P21" << P21 << endl;
        Info << "P22" << P22 << endl;
        Info << "P32" << P32 << endl;
    }

    vector P33 = cmptSqrt(GamSqr - cmptSqr(P31) - cmptSqr(P32) );
    if (debugParticles_) Info << "P33" << P33 << endl;

    scalar randi = rnd.scalarNormal();
    scalar randj = rnd.scalarNormal();
    scalar randk = rnd.scalarNormal();
    vector G1i = vector(randi, randj, randk);

    randi = rnd.scalarNormal();
    randj = rnd.scalarNormal();
    randk = rnd.scalarNormal();
    vector G2i = vector(randi, randj, randk);

    randi = rnd.scalarNormal();
    randj = rnd.scalarNormal();
    randk = rnd.scalarNormal();
    vector G3i = vector(randi, randj, randk);
    if (not diffusionTermOn_)
    {
        G1i = Zero;
        G2i = Zero;
        G3i = Zero;
    }

    p.G1() = G1i;
    p.G2() = G2i;


    if (debugParticles_)
    {
        Info << "random variables : G1 = " << G1i << tab << G2i << endl;
    }
    gammaN =  cmptMultiply(P11, G1i);

    // Stochastic integral 
    OmegaN = cmptMultiply(P21, G1i)
        + cmptMultiply(P22, G2i);
    vector GammaN = cmptMultiply(P31, G1i) + cmptMultiply(P32, G2i) + cmptMultiply(P33, G3i);

    p.UTurb() = (p.UTurbN() * exp(-deltaT * G)
            + (CiTi * (1. - exp(-deltaT * G)))
            + gammaN);

    p.U() = (p.UN() * exp(-deltaT / tauP)
            + cmptMultiply(D, p.UTurbN())
            + cmptMultiply(CiTi, (EminD))
            + GammaN);

    p.stochastic() = true;

    if (debugParticles_)
    {
        Info << "Us^N is " << p.UTurbN() << endl;
        Info << "Us^N+1 is " << p.UTurb() << endl;
        Info << "Up^N is " << p.UN() << endl;
        Info << "Up^N+1 is " << p.U() << endl;
        Info << "Uc^N is " << p.UcTildeN() << endl;
        Info << "Uc^N+1 is " << Uc[p.cell()] << endl;
        Info << "Bi^2 is " << BiHatSqr << endl;
        const vector term1 = cmptMultiply(p.UTurbN(), cmptExp(cmptDivideHW(-deltaT, Ti)));
        const vector term2 = cmptMultiply(CiTi, ones-cmptExp(cmptDivideHW(-deltaT, Ti)));
        const vector term2g = cmptMultiply(g[celli], Ti) * (1.0 - exp(-deltaT / Ti[0]));
        const vector term3 = P11;
        Info << "[dimensional] term 1 is " << term1 << endl;
        Info << "[dimensional] term 2 is " << term2 << endl;
        Info << "[dimensional] term 2 gravity is " << term2g << endl;
        Info << "[dimensional] term 3 is " << term3 << tab << gammaN << endl;

        const vector term1Inertial = p.UN() * exp(-deltaT / tauP);
        const vector term2Inertial = cmptMultiply(D, p.UTurbN());
        const vector term3Inertial = cmptMultiply(CiTi, (EminD));
        const vector termGravityInertial = g[celli] * tauP * E ;
        const vector term4Inertial = GammaN;
        Info << "new up first term " << endl;
        Info << "[dimensional] inertial term 1 is " << term1Inertial << endl;
        Info << "[dimensional] inertial term 2 is " << term2Inertial << endl;
        Info << "[dimensional] inertial term 3 is " << term3Inertial << endl;
        Info << "[dimensional] inertial term 4 is " << term4Inertial << endl;
        Info << "[dimensional] inertial gravity term is " << termGravityInertial << endl;

        Info << "T^{NN} / dt " << T/deltaT << endl;
        Info << "tauP / dt " << tauP / deltaT << endl;
        Info << "tauPStokes / dt " << tauPStokes / deltaT << endl;
        Info << "tauP / T^{NN} " << tauP / T << endl; 
        Info << "tauPStokes / T^{NN} " << tauPStokes / T << endl;
        Info << "theta " << theta << endl;



    }

    if (stationary_)
    {
        p.s() = vector(0.0, 0.0, 0.0);
    }
    else
    {
        // Some coefficients for stochastic integration
        scalar A1 = tauP*(1.0-exp(-deltaT/tauP));
        vector B1 = cmptMultiply(theta,
                cmptMultiply(Ti, (ones - cmptExp(cmptDivideHW(-deltaT, Ti))))
                -A1*ones);
        vector C1 = (ones*deltaT) - (A1*ones) - B1;

        // displacement over t^n -> t^{n+1}
        // Stochastic integral 
        vector OmegaN = cmptMultiply(P21, G1i)
            + cmptMultiply(P22, G2i);

        p.s() = (A1*p.UN()
                + cmptMultiply(B1, p.UTurbN())
                + cmptMultiply(C1,CiTi)
                + g[celli] * tauP * (deltaT - A1)
                + OmegaN);

        if (debugParticles_)
        {
            Info << "UN*deltaT " << p.UN() * deltaT << endl;
            Info << "A1*up_Norm " << A1*p.UN() << endl;
            Info << "B1*us " << cmptMultiply(B1, p.UTurbN()) << endl;
            Info << "C1*uc " <<  cmptMultiply(C1,CiTi) << endl;
            Info << "g * taup * (dt-A) " << g[celli] * tauP * (deltaT - A1) << endl;
            Info << "OmegaN " << OmegaN << endl;
            Info << " timescale " << timescale << endl;
            Info << "displacement " << p.s() << endl;
        }

    }

    if (debugParticles_)
    {
        Info << "displacement " << p.s() << endl;
    }


    p.UNext() = p.U(); // save integrated velocity for next time step
    p.U() = p.UN(); // reset U to UN so if rebounded, UN is changed
    p.stochastic() = true;

}
Info << "Solved SDE..." << endl;

if (checkForExtrapolation) 
{
    Info << "Warning - NN outputs out of distribution - unexpected behaviour " << endl;
}


// ************************************************************************* //


