functions {    
    real sinusoid_func(vector A, vector B, row_vector f, int m, real beta,
                       real sigma, vector dat, vector t, int n)
    {
        vector[n] g;
        vector[n] err;
        real logL;
        g = (cos(t * (2 * pi() * f)) * A) + (sin(t * (2 * pi() * f)) * B);
        err = g - dat;
        logL = -1 * sum(err .* err) / (2 * sigma ^ 2);
        return beta * logL;
    }
}

data {
    int n;              // Number of data points
    int m;              // Number of signals
    vector[n] dat_stan;      // Signal vector
    vector[n] t_stan;        // Time vector
    real Amax;          // Max amplitude
    real Amin;          // Min amplitude
    real freqmax;       // Max frequency
    real freqmin;       // Min frequency
    real sigma;         // Stdev for likelihood
    real beta;          // Inverse temperature
}

parameters {
    vector<lower=0,upper=1>[3*m] alpha;
}

transformed parameters {
    vector[m] A;
    vector[m] B;
    vector[m] ft;
    row_vector[m] f;
    A = (Amax - Amin) * alpha[1:m] + Amin;
    B = (Amax - Amin) * alpha[m+1:2*m] + Amin;
    ft = (freqmax - freqmin) * alpha[2*m+1:3*m] + freqmin;
    f = transpose(ft);
    // for (i in 1:m)
    // {
    //    A[i] = (Amax - Amin) * alpha[(i-1)*3 + 1] + Amin;
    //    B[i] = (Amax - Amin) * alpha[(i-1)*3 + 2] + Amin;
    //    f[i] = (freqmax - freqmin) * alpha[(i-1)*3 + 3] + freqmin;
    // }
}

model {
    alpha ~ uniform(0, 1);
    target += sinusoid_func(A, B, f, m, beta, sigma, dat_stan, t_stan, n);
}

