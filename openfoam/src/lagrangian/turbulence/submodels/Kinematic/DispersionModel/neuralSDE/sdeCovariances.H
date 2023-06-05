template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::gammaLowerSqr(const scalar deltaT, 
                                                     const vector BiSqr, 
                                                     const vector theta,
                                                     const vector Ti, 
                                                     const scalar tau) const
{
    const vector ones = vector(1.,1.,1.);
    const vector BiSqrTi = cmptMultiply(BiSqr, Ti);
    const vector gammaSqr = cmptMultiply(0.5*BiSqrTi, ones-cmptExp(cmptDivideHW(-2.0*deltaT,Ti)));

    return gammaSqr;
}

template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::GammaUpperSqr(const scalar deltaT, 
                                                     const vector BiSqr, 
                                                     const vector theta,
                                                     const vector Ti, 
                                                     const scalar tau) const
{
    const vector thetaSqrBiSqr = cmptMultiply(cmptSqr(theta),BiSqr);
    const vector ones = vector(1.,1.,1.);
    // 0.5 * Ti *(1-exp(-2dt/Ti))
    const vector term1 = cmptMultiply(Ti, 0.5*(ones-cmptExp(cmptDivideHW(-2.0*deltaT,Ti))));
    // 2.0 * tau * Ti / (Ti+tau) * [1-exp(-dt/Ti)*exp(-dt/tau)]
    const vector term2 = 2.0*cmptMultiply( cmptDivideHW(Ti*tau, (Ti+tau*ones)),
                                                ( ones - cmptExp(cmptDivideHW(-deltaT,Ti))*exp(-deltaT/tau) )
                                             );
    // 0.5 * tau * (1 - exp(-2dt/tau) )
    const vector term3 = 0.5*tau*(1.0-exp(-2.0*deltaT/tau))*ones;


    const vector GammaUSqr = cmptMultiply(thetaSqrBiSqr, (term1 - term2 + term3));

    return GammaUSqr;
}

template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::OmegaSqr(const scalar deltaT, 
                                                     const vector BiSqr,
                                                     const vector theta,
                                                     const vector Ti,
                                                     const scalar tau) const
{
    const vector ones = vector(1.,1.,1.);
    // (Ti-tau)^2*dt
    const vector term1 = cmptSqr(Ti-(ones*tau))*deltaT;
    // Ti^3 * 0.5 *[1-exp(-2dt/Ti)]
    const vector term2 = cmptMultiplyHW( 0.5*Ti, cmptSqr(Ti), (ones-cmptExp(cmptDivideHW(-2.0*deltaT,Ti))) );
    // 0.5 * tau^3 * [1-exp(-2dt/tau)]
    const vector term3 = tau*tau*tau*0.5*(1.0 - exp(-2.0*deltaT/tau))*ones;
    // 2 Ti^2 * (Ti-tau) * [1-exp(-dt/Ti)]
    const vector term4 = 2.0*cmptMultiplyHW( cmptSqr(Ti), Ti-(ones*tau), ones - cmptExp(cmptDivideHW(-deltaT,Ti)) );
    // 2 tau^2 * (Ti-tau) * [1-exp(-dt/tau)]
    const vector term5 = 2.0*sqr(tau)*(Ti-(ones*tau))*(1.0 - exp(-deltaT/tau));
    // 2 * tau^2 * [Ti^2/(Ti+tau)] * [1-exp(-dt/Ti)exp(-dt/tau)]
    const vector term6 = 2.0*sqr(tau)*cmptMultiply(cmptDivideHW(cmptSqr(Ti), 
                                                                Ti+(tau*ones)) , 
                                                   ones - cmptExp(cmptDivideHW(-deltaT,Ti))*exp(-deltaT/tau) );

    const vector OmegaSqr = term1 + term2 + term3 - term4 + term5 - term6;

    /*const vector OmegaSqr = cmptSqr(Ti-ones*tau)*deltaT
                            + cmptMultiply( cmptMultiply(Ti, cmptSqr(Ti)),0.5*(ones-cmptExp(cmptDivideHW(-2.0*deltaT*ones,Ti))) ) 
                            + pow(tau,3.0)*0.5*(1.0 - exp(-2.0*deltaT/tau))*ones
                            - 2.0*cmptMultiply( cmptMultiply(cmptSqr(Ti), (Ti-ones*tau)), (ones - cmptExp(cmptDivideHW(-deltaT*ones,Ti))) )
                            + 2.0*sqr(tau)*(Ti-ones*tau)*(1.0 - exp(-deltaT/tau))
                            - (2.0*sqr(tau)*cmptMultiply(cmptDivideHW(cmptSqr(Ti), (Ti+tau*ones)) , (ones - cmptExp(cmptDivideHW(-deltaT*ones,Ti)*exp(-deltaT/tau)) )));
    */
    return OmegaSqr;
}

template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::gammaGamma(const scalar deltaT, 
                                                     const vector BiSqr,
                                                     const vector theta,
                                                     const vector Ti,
                                                     const scalar tau) const
{
    const vector thetaBiSqrTi = cmptMultiplyHW(BiSqr,theta,Ti);
    const vector ones = vector(1.,1.,1.);

    //0.5*(1-exp(-2dt/Ti))
    const vector term1 = 0.5*(ones - cmptExp(cmptDivideHW(-2.0*deltaT,Ti)));
    // [tau/(tau+Ti)]*[1-exp(-dt/Ti)*exp(-dt/tau)]
    const vector term2 = cmptMultiply(cmptDivideHW(tau, 
                                                   (tau*ones)+Ti), 
                                     (ones - cmptExp(cmptDivideHW(-deltaT*ones,Ti))*exp(-deltaT/tau))
                                     );

    const vector gG = cmptMultiply(thetaBiSqrTi,(term1-term2));

/*    const vector gG = cmptMultiply(thetaBiSqrTi,
                                 (0.5*(ones - cmptExp(cmptDivideHW(-2.0*deltaT*ones,Ti))) 
                                    - cmptMultiply(cmptDivideHW(tau*ones, (tau*ones)+Ti), (ones - cmptExp(cmptDivideHW(-deltaT*ones,Ti))*exp(-deltaT/tau)))
                                  ));
*/
    return gG;
}

template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::gammaOmega(const scalar deltaT, 
                                                     const vector BiSqr,
                                                     const vector theta,
                                                     const vector Ti,
                                                     const scalar tau) const
{
    const vector thetaBiSqrTi = cmptMultiplyHW(BiSqr,theta,Ti);
    const vector ones = vector(1.,1.,1.);


    // (Ti-tau)*(1-exp(-dt/Ti))
    const vector term1 = cmptMultiply(Ti-(tau*ones), 
                                      ones-cmptExp(cmptDivideHW(-deltaT,Ti))
                                     );
    // 0.5 Ti (1-exp(-2dt/Ti))
    const vector term2 = 0.5*cmptMultiply(Ti, 
                                          ones-cmptExp(cmptDivideHW(-2.0*deltaT,Ti)) 
                                         );
    // tau^2 / (Ti+tau) * (1-exp(-dt/Ti)exp(-dt/tau))
    const vector term3 = cmptMultiply(cmptDivideHW(sqr(tau), Ti + (tau*ones)), 
                                      ones - cmptExp(cmptDivideHW(-deltaT,Ti))*exp(-deltaT/tau)
                                     );

    const vector gO = cmptMultiply(thetaBiSqrTi, term1-term2+term3);

    // gammaOmega
    return gO;
}

template<class CloudType>
Foam::vector
Foam::neuralSDE<CloudType>::GammaOmega(const scalar deltaT, 
                                                     const vector BiSqr,
                                                     const vector theta,
                                                     const vector Ti,
                                                     const scalar tau) const
{
    // Covariance matrix of [Gamma (Upper case) Omega upper case] 
    const vector ones = vector(1.,1.,1.);
    const vector TiMinTaup = Ti - (tau*ones);
    // Ti [1-exp(-dt/Ti)]
    const vector term1 = cmptMultiply(Ti, ones - cmptExp(cmptDivideHW(-deltaT,Ti)) );
    // tau [ 1-exp(-dt/tau)]
    const vector term2 = tau*ones*(1.0 - exp(-deltaT/tau));
    // 0.5 Ti^2 [1-exp(-2dt/Ti)]
    const vector term3 = 0.5*cmptMultiply( cmptSqr(Ti), ones - cmptExp(cmptDivideHW(-2.0*deltaT,Ti)) );
    // 0.5 tau^2 [1-exp(-2dt/tau)]
    const vector term4 = 0.5*ones*sqr(tau)*(1.0 - exp(-2.0*deltaT/tau));
    // tau Ti [(1-exp(-dt/Ti) exp(-dt/tau)]
    const vector term5 = cmptMultiply(Ti*tau, ones - cmptExp(cmptDivideHW(-deltaT, Ti))*exp(-deltaT/tau) );

    const vector GO = cmptMultiply(TiMinTaup,
                                  ( term1 
                                  - term2))
                                  - term3
                                  - term4
                                  + term5
                                    ;

    // gammaOmega
    return GO;
}



