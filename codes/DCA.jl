using JuMP, Gurobi

landscape1= [0. 0 0 2 2 0 0 0 2 0 0 2;
              2 0 1 2 5 5 0 0 2 0 0 0;
              3 1 2 2 1 1 2 0 0 3 0 0;
              2 1 1 2 1 2 2 0 3 4 4 0
              2 2 1 1 1 1 1 0 3 4 4 3;
              2 2 1 1 1 1 1 1 3 3 3 0;
              0 2 0 1 5 5 5 0 1 1 0 3;
              3 0 0 1 0 0 2 0 0 3 1 3;
              3 4 0 1 1 3 2 2 2 3 3 3;
              4 3 2 0 0 3 2 2 2 3 2 2] + 1

landscape2 = [1. 1 1 1 1 1 3 3   3 3 1 1 1 1 1 1   3 3 1 1 1 1 3 3;
    1 1 1 1 1 1 3 3   3 3 1 1 1 1 1 1   3 3 1 1 1 1 3 3;
    3 3 1 1 2 2 3 3   6 6 6 6 1 1 1 1   3 3 1 1 1 1 1 1;
    3 3 1 1 2 2 3 3   6 6 6 6 1 1 1 1   3 3 1 1 1 1 1 1; 
    4 4 2 2 3 3 3 3   2 2 2 2 3 3 1 1   1 1 4 4 1 1 1 1;
    4 4 2 2 3 3 3 3   2 2 2 2 3 3 1 1   1 1 4 4 1 1 1 1;
    3 3 2 2 2 2 3 3   2 2 3 3 3 3 1 1   4 4 5 5 5 5 1 1;
    3 3 2 2 2 2 3 3   2 2 3 3 3 3 1 1   4 4 5 5 5 5 1 1;
    3 3 3 3 2 2 2 2   2 2 2 2 2 2 1 1   4 4 5 5 5 5 4 4;
    3 3 3 3 2 2 2 2   2 2 2 2 2 2 1 1   4 4 5 5 5 5 4 4;
    
    3 3 3 3 2 2 2 2   2 2 2 2 2 2 2 2   4 4 4 4 4 4 1 1;
    3 3 3 3 2 2 2 2   2 2 2 2 2 2 2 2   4 4 4 4 4 4 1 1;
    1 1 3 3 1 1 2 2   6 6 6 6 6 6 1 1   2 2 2 2 1 1 4 4;
    1 1 3 3 1 1 2 2   6 6 6 6 6 6 1 1   2 2 2 2 1 1 4 4;
    4 4 1 1 1 1 2 2   1 1 1 1 3 3 1 1   1 1 4 4 2 2 4 4;
    4 4 1 1 1 1 2 2   1 1 1 1 3 3 1 1   1 1 4 4 2 2 4 4;
    4 4 5 5 1 1 2 2   2 2 4 4 3 3 3 3   3 3 4 4 4 4 4 4;
    4 4 5 5 1 1 2 2   2 2 4 4 3 3 3 3   3 3 4 4 4 4 4 4;
    5 5 4 4 3 3 1 1   1 1 4 4 3 3 3 3   3 3 4 4 3 3 3 3;
    5 5 4 4 3 3 1 1   1 1 4 4 3 3 3 3   3 3 4 4 3 3 3 3]

Ncell1 = 30
Ncell2 = 80


function create_pixel_mat(landscape, ordered_vec)
    retmat = deepcopy(landscape)
    for i = 1:length(ordered_vec)
        retmat[abs.(retmat - i) .< 0.000000001] = ordered_vec[i]
    end
    
    if size(retmat) == (10,12)
        retmat_zone = hcat(reshape(retmat[1:5 , 1:6],Ncell1),
            reshape(retmat[1:5 , 7:12],Ncell1),
            reshape(retmat[6:10 , 1:6],Ncell1),
            reshape(retmat[6:10 , 7:12],Ncell1))
    else
        retmat_zone = hcat(reshape(retmat[1:10 , 1:8],Ncell2),
            reshape(retmat[1:10 , 9:16],Ncell2),
            reshape(retmat[1:10 , 17:24],Ncell2),
            reshape(retmat[11:20 , 1:8],Ncell2),
            reshape(retmat[11:20 , 9:16],Ncell2),
            reshape(retmat[11:20 , 17:24],Ncell2))
    end
    retmat_zone
end


# parameters that do not change across dataset 1 and 2

maxiter = 1000 # number of iterations for DCA
inc_fac = 1.01 # factor applied to ρ and t as suggested in the paper
ϵ1 = 1e-7
ϵ2 = 1e-7

function DCA(Φ,prior,α,Z,W,w,Ncell)
    S = size(W)[3]
    k = 1
    
    # Set initial ρ and t. As in the paper, ρ and t should not exceed the derived overestimations
    # throughout the iterations
    ρ_UB = max(S*α*(w^2) + S*α*(w^2)*Φ + α*w, 2*Φ*α*(w^2)*(S*Φ) + Ncell*w*α)
    t_UB = Φ * α * (w^2) * (S * Φ) 
    ρ = ρ_UB / (inc_fac^maxiter)
    t = t_UB / (inc_fac^maxiter)
    
    
    x = ones(Ncell,Z,S) * Φ / Ncell
    u = ones(Z,S) * 1 / Z
    new_obj = 1

    # Reshape tensors into [N,Z,S] for broadcast calculation
    u = reshape(u,1,Z,S)
    prior = reshape(prior, Ncell,Z,1)

    converge1 = false
    converge2 = false
    runout = false

    print("start cost: ",sum(squeeze(prior,3) .* exp.(-sum(W .* x .* u,3))) + t * sum(u .* (1 - u)),"\n")
    print("start prob: ",sum(squeeze(prior,3) .* exp.(-sum(W .* x .* u,3))),"\n")
    
    while ~(converge1 | converge2 | runout)
    

        delta_x = - prior .* W .* u .* exp.(-sum(W .* x .* u,3))
        delta_u = squeeze(- sum(prior .* W .* x .* exp.(-sum(W .* x .* u,3)),1),1)
        y = ρ * x .- delta_x
        v = ρ * squeeze(u,1) .- delta_u + t * (2 * squeeze(u,1) .- ones(Z,S))


        m_x = Model(solver=GurobiSolver(OutputFlag = 0))

        @variable(m_x,xvar[1:Ncell,1:Z,1:S])
        @variable(m_x,eu_norm)
        @objective(m_x, Min, ρ/2 * eu_norm - sum(xvar .* y ))
        @constraint(m_x, transpose(reshape(xvar,Ncell * Z * S)) *  reshape(xvar,Ncell * Z * S) <= eu_norm) 
        @constraint(m_x, eu_norm >= 0)
        @constraint(m_x,xvar .>= 0)
        @constraint(m_x,[i=1:Z,j=1:S],sum(xvar[:,i,j]) <= Φ)


        solve(m_x)
        x_new = getvalue(xvar)



        m_u = Model(solver=GurobiSolver(OutputFlag = 0))

        @variable(m_u,uvar[1:Z,1:S])
        @variable(m_u,eu_norm)
        @objective(m_u, Min, ρ/2 * eu_norm  - sum(uvar .* v))
        @constraint(m_u, transpose(reshape(uvar,Z*S)) * reshape(uvar,Z*S) <= eu_norm)
        @constraint(m_u, eu_norm >= 0)
        @constraint(m_u,uvar .>= 0)
        @constraint(m_u,uvar .<= 1)
        @constraint(m_u,[i=1:S],sum(uvar[:,i]) == 1)

        solve(m_u)
        u_new = getvalue(uvar)
        u_new = reshape(u_new, 1, Z, S)


        oldvec = [reshape(x,Ncell * Z * S);reshape(u,Z * S)]
        newvec = [reshape(x_new,Ncell * Z * S);reshape(u_new,Z * S)]


        runout = k >= maxiter
        converge2 = maximum(abs.(newvec - oldvec)) <= ϵ2 * (1 + maximum(abs.(newvec)))
        new_obj = sum(squeeze(prior,3) .* exp.(-sum(W .* x_new .* u_new,3))) + t * sum(u_new .* (1 - u_new))
        old_obj = sum(squeeze(prior,3) .* exp.(-sum(W .* x .* u,3))) + t * sum(u .* (1 - u))
        converge1 = abs(new_obj - old_obj) <= ϵ1 * (1 + abs(new_obj))




        k = k + 1
        ρ = ρ * inc_fac
        t = t * inc_fac

        x = x_new
        u = u_new


        print(k,": ",new_obj)
        print('\n')


    end
    
    [x, u, new_obj]

end


###################################################################################################
# Dataset 1

# parameters that change across dataset 1 and 2

prior_ordered_vec = [0.0085,0.001,0.0115,0.013,0.014,0] # prior of target, order given in the paper
α = maximum(prior_ordered_vec) # max prior probability
prior = create_pixel_mat(landscape1, prior_ordered_vec)
s1 = create_pixel_mat(landscape1, [0.4,0.9,0.3,0.2,0.1,0.8])
s2 = create_pixel_mat(landscape1, [0.5,0.1,0.1,0.7,0.6,0.9])
s3 = create_pixel_mat(landscape1, [0.6,0.1,0.4,0.8,0.7,0.1])
s4 = create_pixel_mat(landscape1, [0.8,0.1,0.6,0.2,0.1,0.7])
s5 = create_pixel_mat(landscape1, [0.5,0.3,0.5,0.4,0.3,0.6])
s6 = create_pixel_mat(landscape1, [0.1,0.5,0.2,0.6,0.5,0.2])

Z = 4 # number of zones
S = 6 # number of sensors
W = reshape(hcat(s1,s2,s3,s4,s5,s6),Ncell1,Z,S) # visibility tensor
w = maximum(W) # max visibility


sum(prior)




Φ_vec = 5:5:40
Φ_vec_len = length(Φ_vec)
obj_vec = zeros(Φ_vec_len,1)

for i=1:Φ_vec_len
    Φ = Φ_vec[i]
    x, u, new_obj = DCA(Φ,prior,α,Z,W,w,Ncell1)
    obj_vec[i] = new_obj
#     print("**************************")
end


paper_obj_vec = [0.818115,0.684438,0.579880,0.496526,0.422119,0.363833,0.323616,0.277450]
hcat(Φ_vec,obj_vec,paper_obj_vec)


###################################################################################################
# Dataset 2

# parameters that change across dataset 1 and 2

prior_ordered_vec = [0.002125,0.00025,0.002875,0.00325,0.0035,0]
prior = create_pixel_mat(landscape2, prior_ordered_vec)
α = maximum(prior_ordered_vec) # max prior probability
s1 = create_pixel_mat(landscape2, [0.4,0.9,0.3,0.2,0.1,0.8])
s2 = create_pixel_mat(landscape2, [0.5,0.1,0.1,0.7,0.6,0.9])
s3 = create_pixel_mat(landscape2, [0.6,0.1,0.4,0.8,0.7,0.1])
s4 = create_pixel_mat(landscape2, [0.8,0.1,0.6,0.2,0.1,0.7])
s5 = create_pixel_mat(landscape2, [0.5,0.3,0.5,0.4,0.3,0.6])
s6 = create_pixel_mat(landscape2, [0.1,0.5,0.2,0.6,0.5,0.2])
s7 = create_pixel_mat(landscape2, [0.4,0.3,0.7,0.2,0.2,0.9])
s8 = create_pixel_mat(landscape2, [0.5,0.5,0.5,0.5,0.3,0.4])
s9 = create_pixel_mat(landscape2, [0.6,0.4,0.2,0.6,0.6,0.6])
s10 = create_pixel_mat(landscape2, [0.9,0.7,0.4,0.1,0.4,0.3])

Z = 6 # number of zones
S = 10 # number of sensors
W = reshape(hcat(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10),Ncell2,Z,S) # visibility tensor
w = maximum(W) # max visibility


sum(prior)

Φ_vec = 5:5:40
Φ_vec_len = length(Φ_vec)
obj_vec2 = zeros(Φ_vec_len,1)

for i=1:Φ_vec_len
    Φ = Φ_vec[i]
    x, u, new_obj = DCA(Φ,prior,α,Z,W,w,Ncell2)
    obj_vec2[i] = new_obj
#     print("**************************")
end

paper_obj_vec2 = [0.906248,0.831474,0.766634,0.706112,0.652908,0.610872,0.564653,0.524267]
hcat(Φ_vec,obj_vec2,paper_obj_vec2)


