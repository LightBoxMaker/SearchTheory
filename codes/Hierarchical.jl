CPU_CORES = 51
addprocs(CPU_CORES - 1);
@everywhere using DistributedArrays, JuMP, Distributions, Ipopt
using StatsBase

@everywhere landscape1= [0. 0 0 2 2 0 0 0 2 0 0 2;
              2 0 1 2 5 5 0 0 2 0 0 0;
              3 1 2 2 1 1 2 0 0 3 0 0;
              2 1 1 2 1 2 2 0 3 4 4 0
              2 2 1 1 1 1 1 0 3 4 4 3;
              2 2 1 1 1 1 1 1 3 3 3 0;
              0 2 0 1 5 5 5 0 1 1 0 3;
              3 0 0 1 0 0 2 0 0 3 1 3;
              3 4 0 1 1 3 2 2 2 3 3 3;
              4 3 2 0 0 3 2 2 2 3 2 2] + 1
@everywhere Ncell = 30


@everywhere function create_pixel_mat(landscape, ordered_vec)
    retmat = deepcopy(landscape)
    for i = 1:length(ordered_vec)
        retmat[abs.(retmat - i) .< 0.000000001] = ordered_vec[i]
    end
    
    if size(retmat) == (10,12)
        retmat_zone = hcat(reshape(retmat[1:5 , 1:6],Ncell),
            reshape(retmat[1:5 , 7:12],Ncell),
            reshape(retmat[6:10 , 1:6],Ncell),
            reshape(retmat[6:10 , 7:12],Ncell))
    end
    retmat_zone
end


# Using the prior and visibility info given in the DCA paper
# These have @everywhere because they are fixed. Better declare them in each process rather than
# copying from the parent process
@everywhere s1 = create_pixel_mat(landscape1, [0.4,0.9,0.3,0.2,0.1,0.8])
@everywhere s2 = create_pixel_mat(landscape1, [0.5,0.1,0.1,0.7,0.6,0.9])
@everywhere s3 = create_pixel_mat(landscape1, [0.6,0.1,0.4,0.8,0.7,0.1])
@everywhere s4 = create_pixel_mat(landscape1, [0.8,0.1,0.6,0.2,0.1,0.7])
@everywhere s5 = create_pixel_mat(landscape1, [0.5,0.3,0.5,0.4,0.3,0.6])
@everywhere s6 = create_pixel_mat(landscape1, [0.1,0.5,0.2,0.6,0.5,0.2])




@everywhere Z = 4 # number of zones
@everywhere S = 6 # number of sensors
@everywhere W = reshape(hcat(s1,s2,s3,s4,s5,s6),Ncell,Z,S) # visibility tensor



@everywhere θ = 0.3
@everywhere N = 50
@everywhere Threshold = convert(Int64,floor(N * θ)) # for accepting P computed from each sample
@everywhere epsilon = 1e-5


# No @everywhere because these are only used by the parent process
maxiter = 15 # number of iterations for 



# No @everywhere because these are changing, pass into functions as parameters

Φ = 5
prior_ordered_vec = [0.0085,0.001,0.0115,0.013,0.014,0] # prior of target, order given in the paper
prior = create_pixel_mat(landscape1, prior_ordered_vec)
print(sum(prior))











####################################
## CE Algorithm parallel N

@everywhere function solve_alloc(u, Φ,prior,W)

    m_ϕ = Model(solver = IpoptSolver(print_level=0))

    @variable(m_ϕ,ϕvar[1:Ncell,1:Z,1:S]) # resource allocation for each senor in each zone in each cell
    @variable(m_ϕ,exponential[1:Ncell,1:Z])
    @variable(m_ϕ, aux[1:Ncell,1:Z])
    @objective(m_ϕ, Min, sum(prior .* exponential))
    @NLconstraint(m_ϕ, [i=1:Ncell,j=1:Z], exp(aux[i,j]) <= exponential[i,j])
    @constraint(m_ϕ,aux .== squeeze(-sum(W .* ϕvar .* reshape(u,1,Z,S),3),3))
    @constraint(m_ϕ, exponential .>= 0)
    @constraint(m_ϕ,ϕvar .>= 0)
    @constraint(m_ϕ,[i=1:Z,j=1:S],sum(ϕvar[:,i,j]) <= Φ)

    solve(m_ϕ)
    getobjectivevalue(m_ϕ), transpose(u)
end
    
    

function CE_parallel(Φ,prior,W)
    
    P = ones(S,Z)/Z
    ind = true
    P_holder = zeros(S,Z)
    for iter=1:maxiter

        cumsumP = cumsum(P,2)
        
        vec_tile = rand(Uniform(0,1), S * N,1)
        cumsumP_tile = repmat(cumsumP,N)
        which_zone = floor.(findmax(cumsumP_tile .> vec_tile,2)[2] / (S*N) - epsilon) + 1
        u = abs.(repmat(transpose(1:Z),S*N,1) .- which_zone) .< epsilon
        u = transpose(u)
        
        arr_obj = []
        arr_assign = zeros(S,Z,N)
#         tic()
        retobj = @DArray [solve_alloc(u[:,(i-1)*S+1:i*S],Φ,prior,W) for i = 1:N]
#         toc()
        for i = 1:N
            push!(arr_obj, retobj[i][1])
            arr_assign[:,:,i] = retobj[i][2]
        end
 
        best_ind = sortperm(arr_obj)[1:Threshold]
        P = squeeze(sum(arr_assign[:,:,best_ind],3)/Threshold,3)
        
        if sum(abs.(P - 1) .<=0.000001) == S
#             print(iter,'\n')
            break
        end
    end
    
    u = transpose(P)
    m_ϕ = Model(solver = IpoptSolver(print_level=0))

    @variable(m_ϕ,ϕvar[1:Ncell,1:Z,1:S])
    @variable(m_ϕ,exponential[1:Ncell,1:Z])
    @variable(m_ϕ, aux[1:Ncell,1:Z])
    @objective(m_ϕ, Min, sum(prior .* exponential))
    @NLconstraint(m_ϕ, [i=1:Ncell,j=1:Z], exp(aux[i,j]) <= exponential[i,j])
    @constraint(m_ϕ,aux .== squeeze(-sum(W .* ϕvar .* reshape(u,1,Z,S),3),3))
    @constraint(m_ϕ, exponential .>= 0)
    @constraint(m_ϕ,ϕvar .>= 0)
    @constraint(m_ϕ,[i=1:Z,j=1:S],sum(ϕvar[:,i,j]) <= Φ)

    solve(m_ϕ)
    
    getobjectivevalue(m_ϕ), P, exp.(getvalue(aux)), getvalue(ϕvar)
    
    
    
end



###################################################################################
### Nondetection under the setting of moving target
###################################################################################
moving_mat= [0.0 0.0 0.0;
             0.0 0.3 0.1;
             0.0 0.1 0.5]
sum(moving_mat)
c2c_tensor = construct_c2c(moving_mat)


# construct a 30x4x30x4 one step moving probability tensor
# first, construct 120x120 one step transition matrix

function construct_c2c(moving_mat)
    # here we number all cells from 1 to 120 in a row-major fashion
    c2c = zeros(Ncell * Z, Ncell * Z)
    for i = 1:(Ncell * Z)
        # Stay in current cell
        c2c[i,i] = moving_mat[2,2]

        # move right
        if mod(i,12) ==  0 # right end 
            c2c[i,i] = c2c[i,i] + moving_mat[2,3]
        else
            c2c[i,i+1] = moving_mat[2,3]
        end

        # move down
        if i > 12 * (10-1) # bottom end
            c2c[i,i] = c2c[i,i] + moving_mat[3,2]
        else
            c2c[i,i+12] = moving_mat[3,2]
        end

        # move diagonal
        if (mod(i,12) == 0) | (i > 12 * (10-1))
            c2c[i,i] = c2c[i,i] + moving_mat[3,3]
        else
            c2c[i,i+12+1] = moving_mat[3,3]
        end


    end

    # convert the 120x120 matrix into 30x4x30x4 tensor
    # Note that Julia is column major: suppose the first (leftmost, upmost) cell
    # is cell 1, cell 2 refers to the cell below cell 1 not the cell to the right
    # This affects how cell index is assigned
    c2c_tensor = zeros(Ncell,Z,Ncell,Z)
    for i=1:Ncell*Z
        zone_prev = (i / 60 > 1) * 2 + ((mod(i,12) / 6 > 1) | (mod(i,12) == 0)) + 1
        offset_prev = [0,6,60,66][zone_prev]
        cell_prev = (rem(i-offset_prev,12) - 1) * 5 + div(i-offset_prev,12) + 1 # column major in julia
        for j=1:Ncell*Z
            zone_cur = (j / 60 > 1) * 2 + ((mod(j,12) / 6 > 1) | (mod(j,12) == 0)) + 1
            offset_cur = [0,6,60,66][zone_cur]
            cell_cur = (rem(j-offset_cur,12) - 1) * 5 + div(j-offset_cur,12) + 1 # column major in julia
            c2c_tensor[cell_prev,zone_prev,cell_cur,zone_cur] = c2c[i,j]
        end
    end
    
    return c2c_tensor
end

T = 4
Φ = 5
Total_Iter = 4 # the number of iterations to go from myopic search plan to optimal search plan

################## Takes a while to run ########################

function mov_target_CE()
    DTensor = ones(Ncell,Z,T)
    moving_target_obj_vec = zeros(Total_Iter,T)
    prob_nondetect_Tensor = zeros(Ncell,Z,T)
    final_assignment = zeros(S,Z,T)
    final_res_alloc = zeros(Ncell,Z,S,T)
    
    
    for i = 1:Total_Iter
        # Initialize UTensor
        UTensor = zeros(Ncell,Z,T)
        UTensor[:,:,1] =  prior


        for τ = 1:4
            if τ > 1
                UTensor[:,:,τ] = sum(sum(c2c_tensor .* 
                        reshape(prob_nondetect_Tensor[:,:,τ-1],Ncell,Z,1,1) .* 
                        reshape(UTensor[:,:,τ-1],Ncell,Z,1,1),1),2)
            end
            β = UTensor[:,:,τ] .* DTensor[:,:,τ]
            obj, assignment, prob_nondetect, res_alloc = CE_parallel(Φ,β)
            final_assignment[:,:,τ] = assignment
            final_res_alloc[:,:,:,τ] = res_alloc
            prob_nondetect_Tensor[:,:,τ] = prob_nondetect
            moving_target_obj_vec[i,τ] = obj
            print("Iter: ",i,". Completed timestep: ",τ,". Obj: ",obj,"\n")
        end

        # Compute the DTensors using the nondetection probabilities at each timestep as calculated
        # in the previous iteration
        for τ = T-1:-1:1
            DTensor[:,:,τ] = sum(sum(c2c_tensor .* 
                    reshape(prob_nondetect_Tensor[:,:,τ+1],1,1,Ncell,Z) .* 
                    reshape(DTensor[:,:,τ+1],1,1,Ncell,Z),3),4)
        end
    end
    
    return moving_target_obj_vec, prob_nondetect_Tensor, final_assignment, final_res_alloc

end


tic()
moving_target_obj_vec, prob_nondetect_Tensor, final_assignment, final_res_alloc = mov_target_CE()
toc()



