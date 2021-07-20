from brian2.units import *
Neuron_model = {'model':
                '''
            dImem/dt = (((Ith_clip / Itau_clip) * (Iin_clip  + Ia_clip - Ishunt_clip - Iahp_clip)) - Ith_clip - ((1 + ((Ishunt_clip + Iahp_clip - Ia_clip) / Itau_clip)) * Imem)) / (tau * ((Ith_clip/(Imem + Io)) + 1)) : amp (unless refractory)

          dIahp/dt = (- Ithahp_clip - Iahp + 2*Io*(Iahp<=Io)) / (tauahp * (Ithahp_clip / Iahp + 1)) : amp # adaptation current
        Ia = Iagain / (1 + exp(-(Imem - Iath) / Ianorm)) : amp  # postive feedback current
          Iahp_clip = Iahp*(Imem>Io) + Io*(Imem<=Io)  : amp

            Itau_clip = Itau*(Imem>Io) + Io*(Imem<=Io)  : amp
            Ith_clip = Ith*(Imem>Io) + Io*(Imem<=Io)    : amp
            Iin_clip = clip(Iin+Iconst,Io, 1*amp) : amp
        Ia_clip = Ia*(Imem>Io) + 2*Io*(Imem<=Io)    : amp
            Ithahp_clip = Ithahp*(Iahp>Io) + Io*(Iahp<=Io) : amp
            Ishunt_clip = clip(Ishunt, Io, Imem) : amp

            tau = (Cmem * Ut) / (kappa * Itau_clip) : second        # Membrane time constant
            kappa = (kn + kp) / 2 : 1

            Inoise  : amp                                    # Noise due to mismatch

            kn      : 1 (constant)                   # subthreshold slope factor for nFETs
            kp      : 1 (constant)                   # subthreshold slope factor for pFETs
            Ut      : volt (constant)                # Thermal voltage
            Io      : amp (constant)                 # Dark current
            Cmem    : farad (constant)               # Membrane capacitance
            Ispkthr : amp (constant)                         # Spiking threshold
            Ireset  : amp (constant)                 # Reset current
            refP    : second    (constant)           # refractory period (It is still possible to set it to False)
            Ith     : amp (constant)                         # DPI threshold (low pass filter).
            Itau    : amp (constant)                         # Leakage current
            Iconst  : amp (constant)                         # Additional input current similar to constant current injection
            Ishunt  : amp (constant)                         # Shunting inhibitory current (directly affects soma)
            Ica     : amp (constant)
         


          tauahp = (Cahp * Ut) / (kappa * Itauahp) : second # time constant of adaptation
          Iahpmax = (Ica / Itauahp) * Ithahp_clip : amp     # Ratio of currents through diffpair and adaptation block
          Ithahp : amp (constant)
          Itauahp : amp (constant)
          Cahp : farad (constant)
         


        Iagain : amp (constant)
        Iath : amp (constant)
        Ianorm : amp (constant)

         
           x : 1         (constant)        # x location on 2d grid (only set it if you need it)
           y : 1         (constant)        # y location on 2d grid
           

Iin = Iin0 : amp # input currents
        Iin0 : amp
''',
                'threshold':
                '''Imem > Ispkthr''',
                'reset':
                '''Imem = Ireset;
                  Iahp += Iahpmax;
                  ''',
                'parameters':
                {
                    'Inoise': '0.5 * pamp',
                    'Iconst': '0.5 * pamp',
                    'kn': '0.75',
                    'kp': '0.66',
                    'Ut': '25. * mvolt',
                    'Io': '0.5 * pamp',
                    'Cmem': '1.5 * pfarad',
                    'Iath': '0.5 * namp',
                    'Iagain': '50. * pamp',
                    'Ianorm': '10. * pamp',
                    'Ica': '2. * pamp',
                    'Itauahp': '1. * pamp',
                    'Ithahp': '1. * pamp',
                    'Cahp': '1. * pfarad',
                    'Iahp': '0.5 * pamp',
                    'Ishunt': '0.5 * pamp',
                    'Ispkthr': '1. * namp',
                    'Ireset': '0.6 * pamp',
                    'Ith': '0.9 * pamp',
                    'Itau': '8. * pamp',
                    'refP': '1. * msecond',
                }
                }
