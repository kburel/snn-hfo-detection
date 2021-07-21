from brian2.units import *
Synapse_model = {'model':
                 '''
        dI_syn/dt =  (- I_syn - I_gain + 2*Io_syn*(I_syn<=Io_syn))/(tausyn*((I_gain/I_syn)+1)) : amp (clock-driven)



        Iin{input_number}_post = I_syn *  sign(weight)  : amp (summed)

        weight : 1
        w_plast : 1


        I_gain = Io_syn*(I_syn<=Io_syn) + I_th*(I_syn>Io_syn) : amp


        Itau_syn = Io_syn*(I_syn<=Io_syn) + I_tau*(I_syn>Io_syn) : amp

        baseweight : amp (constant)     # synaptic gain
        tausyn = Csyn * Ut_syn /(kappa_syn * Itau_syn) : second
        kappa_syn = (kn_syn + kp_syn) / 2 : 1

        Iw = abs(weight) * baseweight  : amp

        I_tau        : amp (constant)
        I_th         : amp (constant)
        kn_syn       : 1 (constant)
        kp_syn       : 1 (constant)
        Ut_syn       : volt (constant)
        Io_syn       : amp (constant)
        Csyn         : farad (constant)
          ''',
                 'on_pre':
                 '''

        I_syn += Iw * w_plast * I_gain / (Itau_syn * ((I_gain/I_syn)+1))

          ''',
                 'on_post':
                 '''
   
''',
                 'parameters':
                 {
                     'Io_syn': '0.5 * pamp',
                     'kn_syn': '0.75',
                     'kp_syn': '0.66',
                     'Ut_syn': '25. * mvolt',
                     'Csyn': '1.5 * pfarad',
                     'I_tau': '10. * pamp',
                     'I_th': '10. * pamp',
                     'I_syn': '0.5 * pamp',
                     'w_plast': '1',
                     'baseweight': '7. * pamp',
                 }
                 }
