import numpy as np

def E_step(data, Lambda, Psi):

    features, latent = Lambda.shape
    samples = data.shape[1]
    #  Compute E[z] via eqn 2 in Ghahramani & Hinton (1996)
    # First compute Beta
    Beta = Lambda.T @ np.linalg.inv(Psi + Lambda @ Lambda.T)
    assert Beta.shape == (latent, features)
    # Second compute E_z
    E_z = Beta @ data
    assert E_z.shape == (latent, samples)

    # Compute E[zz^T] via eqn 4 in Ghahramani & Hinton (1996)
    # This involves 3 terms, the first two do not depend on the data
    # so let's compute them first
    term1 = np.eye(latent) 
    term2 = Beta @ Lambda 
    assert term1.shape == term2.shape == (latent, latent)

    E_zz = []
    for i in range(data.shape[1]):
        x = data[:,i, np.newaxis]
        term3 = Beta @ x @ x.T @ Beta.T
        assert term3.shape == (latent, latent)
        E_zz.append(term1 - term2 + term3)
    E_zz = np.array(E_zz)
    assert E_zz.shape == (samples,latent,latent)

    return E_z, E_zz

def M_step(data, E_z, E_zz, Lambda):
    features, latent = Lambda.shape
    samples = data.shape[1]

    # Compute Lambda_new via eqn 5 in Ghahramani & Hinton (1996), which has 2 terms involving summations over samples
    # let's compute each term individually then multiply them together
    Lambda_new_1 = []
    for i in range(data.shape[1]):
        x = data[:,i, np.newaxis] # np.newaxis is used to ensure x is a column vector
        E_zi = E_z[:,i, np.newaxis]
        assert E_zi.shape == (latent,1)
        assert x.shape == (features,1)
        Lambda_new_1.append(x @ E_zi.T)
    Lambda_new_1 = np.array(Lambda_new_1)
    assert Lambda_new_1.shape == (samples,features,latent)
    # sum across samples
    Lambda_new_1 = np.sum(Lambda_new_1, axis=0)
    assert Lambda_new_1.shape == (features, latent)

    # sum E_zz across samples
    Lambda_new_2 = np.linalg.inv(np.sum(E_zz, axis=0))
    assert Lambda_new_2.shape == (latent, latent)

    Lambda_new = Lambda_new_1 @ Lambda_new_2
    assert Lambda_new.shape == (features, latent)


    # Compute Psi_new via eqn 6 in Ghahramani & Hinton (1996), which has a summation over samples that invovles 2 terms
    Psi_new_list = []
    for i in range(data.shape[1]):
        x = data[:,i, np.newaxis]
        E_zi = E_z[:,i, np.newaxis]
        term1 = x @ x.T
        term2 = Lambda_new @ E_zi @ x.T
        assert term1.shape == term2.shape == (features, features)
        Psi_new_list.append(term1 - term2)

    # sum across samples
    Psi_sum = np.sum(np.array(Psi_new_list), axis=0)
    assert Psi_sum.shape == (features,features)

    # Extract the diagonal elements, divide by number of samples
    diag = np.diag(Psi_sum)/samples
    # reconstruct Psi_new with zeros off the diagonal
    Psi_new = np.zeros_like(Psi_sum)
    np.fill_diagonal(Psi_new, diag)
    assert Psi_new.shape == (features, features)

    return Lambda_new, Psi_new

# EM Algorithm
def em_factor_analysis(data, Lambda, Psi, tol=0.001, max_iter=1000):

    # collect the change in Lambda and Psi for each iteration
    delta_Lambda_list = []
    delta_Psi_list = []

    for iter in range(max_iter):
        # E-step: compute E_z and E_zzT for each data point i
        E_z, E_zz = E_step(data, Lambda, Psi)
        
        # M-step: update Lambda and Psi
        Lambda_new, Psi_new = M_step(data, E_z, E_zz, Lambda)

        delta_Lambda = np.abs(np.linalg.norm(Lambda_new - Lambda))
        delta_Psi = np.abs(np.linalg.norm(Psi_new - Psi))

        delta_Lambda_list.append(delta_Lambda)
        delta_Psi_list.append(delta_Psi)

        if delta_Lambda < tol and delta_Psi < tol:
            break

        Lambda = Lambda_new
        Psi = Psi_new
        # likelihood_old = likelihood_new

    # return Lambda, Psi
    return Lambda, Psi, delta_Lambda_list, delta_Psi_list
          
