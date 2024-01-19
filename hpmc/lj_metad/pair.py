lj = '''
float rsq = dot(r_ij, r_ij);
const float sigma = {sigma:.15f};
const float sigma_sq = sigma*sigma;
const float a=sigma_sq/rsq;
const float epsilon = {epsilon:.15f};

if (rsq < 3.0f*sigma)
    return 4 * epsilon * ( a*a*a*a*a*a - a*a*a );
else
    return 0.0f;
'''
