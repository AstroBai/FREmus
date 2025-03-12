import torch
import torch.nn as nn
import pickle
import numpy as np
from scipy.interpolate import CubicSpline
import os
import camb

class BkANN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(BkANN, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1)
        self.hidden_activation1 = nn.Sigmoid()
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.hidden_activation2 = nn.Sigmoid()
        self.output_layer = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_activation1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_activation2(x)
        x = self.output_layer(x)
        return x
    
class emulator:
    def __init__(self):
        
        module_path = os.path.dirname(__file__)
        cache_path = os.path.join(module_path, 'cache_v1') 
        cache_boost_path = os.path.join(module_path, 'cache_boost') 
        self.ks = np.logspace(-4,0,128)
        self.scaler = None
        with open(os.path.join(cache_path,'scaler.pkl'), 'rb') as scaler_file:
            self.scaler = pickle.load(scaler_file)
        
        self.pc = {}
        self.mean = {}
        
        for z in [0.0, 0.5, 1.0, 2.0, 3.0]:
            pc = np.load(os.path.join(cache_path,f'pc_{z:.1f}.npy'))
            self.mean[z] = pc[0, :]
            self.pc[z] = pc[1:, :]
        n_hidd = 256
        n_out = 24
        self.model0 = BkANN(7, n_hidd, n_hidd, n_out)
        self.model05 = BkANN(7, n_hidd, n_hidd, n_out)
        self.model1 = BkANN(7, n_hidd, n_hidd, n_out)
        self.model2 = BkANN(7, n_hidd, n_hidd, n_out)
        self.model3 = BkANN(7, n_hidd, n_hidd, n_out)
            
        self.model0.load_state_dict(torch.load(os.path.join(cache_path,'ann_0.0.pth')))
        self.model05.load_state_dict(torch.load(os.path.join(cache_path,'ann_0.5.pth')))
        self.model1.load_state_dict(torch.load(os.path.join(cache_path,'ann_1.0.pth')))
        self.model2.load_state_dict(torch.load(os.path.join(cache_path,'ann_2.0.pth')))
        self.model3.load_state_dict(torch.load(os.path.join(cache_path,'ann_3.0.pth')))
        
        
        # for boost
        with open(os.path.join(cache_boost_path,'scaler.pkl'), 'rb') as scaler_file:
            self.scaler_boost = pickle.load(scaler_file)
        
        self.pc_boost = {}
        self.mean_boost = {}
        
        for z in [0.0, 0.5, 1.0, 2.0, 3.0]:
            pc = np.load(os.path.join(cache_boost_path,f'pc_{z:.1f}.npy'))
            self.mean_boost[z] = pc[0, :]
            self.pc_boost[z] = pc[1:, :]
        n_hidd = 256
        n_out = 24
        self.boost_model0 = BkANN(7, n_hidd, n_hidd, n_out)
        self.boost_model05 = BkANN(7, n_hidd, n_hidd, n_out)
        self.boost_model1 = BkANN(7, n_hidd, n_hidd, n_out)
        self.boost_model2 = BkANN(7, n_hidd, n_hidd, n_out)
        self.boost_model3 = BkANN(7, n_hidd, n_hidd, n_out)
            
        self.boost_model0.load_state_dict(torch.load(os.path.join(cache_boost_path,'ann_0.0.pth')))
        self.boost_model05.load_state_dict(torch.load(os.path.join(cache_boost_path,'ann_0.5.pth')))
        self.boost_model1.load_state_dict(torch.load(os.path.join(cache_boost_path,'ann_1.0.pth')))
        self.boost_model2.load_state_dict(torch.load(os.path.join(cache_boost_path,'ann_2.0.pth')))
        self.boost_model3.load_state_dict(torch.load(os.path.join(cache_boost_path,'ann_3.0.pth')))

    def get_k_values(self):
        return self.ks

    def set_cosmo(self, Om=0.3, Ob=0.05, h=0.7, ns=1.0, mnu=0.05, fR0=-3e-5, As=2e-9, redshifts=[3.0,2.0,1.0,0.5,0.0],use_emu=False):
        """
        
        Set cosmology parameters for the emulator.
        Parameters:
        Om (float): Omega matter
        Ob (float): Omega baryon
        h (float): Hubble constant
        ns (float): scalar spectral index
        mnu (float): sum of neutrino masses
        fR0 (float): dark energy equation of state at z=0
        As (float): amplitude of primordial power spectrum
        redshifts (list): list of redshifts for which to compute power spectra
        use_emu (bool): whether to use pure emulator or Fid-Boost way to compute power spectra. Default is False, i.e. use Fid-Boost.
        
        """
        self.Om = Om
        self.Ob = Ob
        self.h = h
        self.ns = ns
        self.As = As
        self.mnu = mnu
        self.fR0 = fR0
        self.Onu = mnu / 93.14 / h**2
        self.use_emu = use_emu
        if use_emu is not True:
            # camb
            Onu = mnu / 93.14 / h**2
            camb_pars = camb.CAMBparams(WantCls=False,DoLensing=False)
            camb_pars.NonLinearModel.set_params(halofit_version='mead2020')
            camb_pars.set_cosmology(H0=h * 100, ombh2=Ob * h**2, omch2=(Om - Ob - Onu) * h**2, mnu=mnu, omk=0, num_massive_neutrinos=3)
            camb_pars.InitPower.set_params(ns=ns, As=As)
            camb_pars.set_matter_power(redshifts=[3.0, 2.0, 1.0, 0.5, 0.0], kmax=10, nonlinear=True)
            results = camb.get_results(camb_pars)
            self.mpi = results.get_matter_power_interpolator(nonlinear=True)
        
    
    def get_boost(self, k=None, z=None, return_k_values=False):
        """
        
        Get boost factor for a given k and redshift.
        Parameters:
        k (array): wavenumbers at which to compute boost factor
        z (float): redshift at which to compute boost factor
        return_k_values (bool): whether to return k values along with boost factor. Default is False.

        """
        
        if z is None:
            z = 0
            print('WARNING: No redshift value given, the default value is z=0.0')

        if k is None:
            k = self.ks
    
        try:
            # Step 1: Prepare parameters for ANN
            params = np.array([self.Om, self.Ob, self.h, self.ns, self.mnu, self.fR0, self.As])
            X = params.reshape(1, -1)
            X = self.scaler_boost.transform(X)
            X_tensor = torch.tensor(X, dtype=torch.float32)
        
            # Step 2: Identify redshift range for ANN
            z_range = None
            for range_ in [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0)]:
                if range_[0] <= z <= range_[1]:
                    z_range = range_
                    break

            if z_range is None:
                raise ValueError('Redshift should be between 0.0 and 3.0')
        
            # Step 3: Get ANN predictions for Bk
            pc0 = self.pc_boost[0.0]
            pc05 = self.pc_boost[0.5]
            pc1 = self.pc_boost[1.0]
            pc2 = self.pc_boost[2.0]
            pc3 = self.pc_boost[3.0]
            mean0 = self.mean_boost[0.0]
            mean05 = self.mean_boost[0.5]
            mean1 = self.mean_boost[1.0]
            mean2 = self.mean_boost[2.0]
            mean3 = self.mean_boost[3.0]
        
            with torch.no_grad():
                w_tensor0 = self.boost_model0(X_tensor)
                w_tensor05 = self.boost_model05(X_tensor)
                w_tensor1 = self.boost_model1(X_tensor)
                w_tensor2 = self.boost_model2(X_tensor)
                w_tensor3 = self.boost_model3(X_tensor)
            
                Bk0 = np.dot(w_tensor0.numpy(), pc0) + mean0
                Bk05 = np.dot(w_tensor05.numpy(), pc05) + mean05
                Bk1 = np.dot(w_tensor1.numpy(), pc1) + mean1
                Bk2 = np.dot(w_tensor2.numpy(), pc2) + mean2
                Bk3 = np.dot(w_tensor3.numpy(), pc3) + mean3
            
            Bk = CubicSpline(np.array([0.0, 0.5, 1.0, 2.0, 3.0]), np.array([Bk0, Bk05, Bk1, Bk2, Bk3]), axis=0)
            Bk = Bk(z)
            Bk = Bk.reshape(-1, 1)
            Bk_interp = CubicSpline(self.ks, Bk)
            bk_mg = Bk_interp(k)
            if return_k_values:
                bk_mg = np.column_stack((k, bk_mg))
                
            return bk_mg.ravel()
        except ValueError as e:
            print(e)


    
    def get_power_spectrum(self, k=None, z=None, return_k_values=False):
        
        """
        
        Get matter power spectrum for a given k and redshift.
        Parameters:
        k (array): wavenumbers at which to compute power spectrum
        z (float): redshift at which to compute power spectrum
        return_k_values (bool): whether to return k values along with power spectrum. Default is False.
        
        """
        
        if z is None:
            z = 0
            print('WARNING: No redshift value given, the default value is z=0.0')

        if k is None:
            k = self.ks

        if self.use_emu:
        
            try:
                # Step 1: Prepare parameters for ANN
                params = np.array([self.Om, self.Ob, self.h, self.ns, self.mnu, self.fR0, self.As])
                X = params.reshape(1, -1)
                X = self.scaler.transform(X)
                X_tensor = torch.tensor(X, dtype=torch.float32)
            
                # Step 2: Identify redshift range for ANN
                z_range = None
                for range_ in [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0)]:
                    if range_[0] <= z <= range_[1]:
                        z_range = range_
                        break

                if z_range is None:
                    raise ValueError('Redshift should be between 0.0 and 3.0')
            
                # Step 3: Get ANN predictions for Bk
                pc0 = self.pc[0.0]
                pc05 = self.pc[0.5]
                pc1 = self.pc[1.0]
                pc2 = self.pc[2.0]
                pc3 = self.pc[3.0]
                mean0 = self.mean[0.0]
                mean05 = self.mean[0.5]
                mean1 = self.mean[1.0]
                mean2 = self.mean[2.0]
                mean3 = self.mean[3.0]
            
                with torch.no_grad():
                    w_tensor0 = self.model0(X_tensor)
                    w_tensor05 = self.model05(X_tensor)
                    w_tensor1 = self.model1(X_tensor)
                    w_tensor2 = self.model2(X_tensor)
                    w_tensor3 = self.model3(X_tensor)
                
                    Pk0 = np.dot(w_tensor0.numpy(), pc0) + mean0
                    Pk05 = np.dot(w_tensor05.numpy(), pc05) + mean05
                    Pk1 = np.dot(w_tensor1.numpy(), pc1) + mean1
                    Pk2 = np.dot(w_tensor2.numpy(), pc2) + mean2
                    Pk3 = np.dot(w_tensor3.numpy(), pc3) + mean3
                
                Pk = CubicSpline(np.array([0.0, 0.5, 1.0, 2.0, 3.0]), np.array([Pk0, Pk05, Pk1, Pk2, Pk3]), axis=0)
                Pk = Pk(z)
                Pk = Pk.reshape(-1, 1)
                Pk_interp = CubicSpline(self.ks, Pk)
                pk_mg = 10 ** Pk_interp(k)
                if return_k_values:
                    pk_mg = np.column_stack((k, pk_mg))

                return pk_mg.ravel()
            except ValueError as e:
                print(e)
                
        else:
            pk_fid = self.mpi.P(z,k)        
            bk = self.get_boost(k=k, z=z, return_k_values=False)
            pk_mg = pk_fid * bk
            if return_k_values:
                pk_mg = np.column_stack((k, pk_mg))
            return pk_mg.ravel()