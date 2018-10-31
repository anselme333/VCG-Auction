# VCG auction model
# Author: Anselme
# Interpreter: python 2.7.14 and julia-0.5.2
################################################################################
# Needed packages
using JuMP
using PyPlot
using StatsBase
using Distributions
using CSV

################################################################################
# Inputs
starting_time= Dates.now()
RANDOM_SEED = 1000
srand(RANDOM_SEED)
num_MVNO = 10
max_iter = num_MVNO
# The vector of  bidding values per unit of cache storage
bid_MVNO = rand(Uniform(10,20),num_MVNO)
# The vector of cache demands from MVNOs
MVNO_Cache_Req = rand(Uniform(60,200), num_MVNO)
# Total cache capacity of infrastructure provider
InP_max_cache_capacity =400

# Reserved price per unit of cache storage
reserved_cache_price=15


eligible_MVNO = zeros(0)
eligiblle_demands_mvno = zeros(0)
println("Initial bidding values", bid_MVNO)
i=1
while i <= length(bid_MVNO)
	if bid_MVNO[i]>= reserved_cache_price
	eligible_MVNO = append!(eligible_MVNO, bid_MVNO[i])
	eligiblle_demands_mvno =append!(eligiblle_demands_mvno, MVNO_Cache_Req[i])
 end
 i += 1
 end
println("eligible MVNO ",  eligible_MVNO)
println("eligible demands ", eligiblle_demands_mvno)

# Just keep the copy of eligible bidding values and demands
eligible_bidding_value =abs(eligible_MVNO)
eligible_bidding_demand =abs(eligiblle_demands_mvno)
total_cache_capaciy= abs(InP_max_cache_capacity)

###############################################################################
# Initialization
# The welfare of other players than MVNO m from the chosen outcome
# when MVNO m participates in the auction
v_m  = zeros(0)

# The welfare for other bidders than MVNO m from the chosen outcome when MVNO
# m is not participating in the auction
v_no_m = zeros(0)

# Winning  bidding values
w = zeros(0)

#  Optiaml Payment

P_optimal = zeros(size(eligible_MVNO))

#  Auction decision variable

x_decision = zeros(size(eligible_MVNO))
###############################################################################
# Winner determination
while length(eligible_MVNO) > 0 && length(eligiblle_demands_mvno) > 0
			 max_bid_value = maximum(eligible_MVNO)   # Find maximum bidding values
			 k =findfirst(eligible_MVNO,max_bid_value)
			 if eligiblle_demands_mvno[k] <= InP_max_cache_capacity
				w=append!(w, max_bid_value)
				end
				InP_max_cache_capacity = InP_max_cache_capacity - eligiblle_demands_mvno[k]
				eligible_MVNO = deleteat!(eligible_MVNO,k)
				eligiblle_demands_mvno = deleteat!(eligiblle_demands_mvno,k)
end
println("winnner value", w)


# just keep a copy of original values
winning_value = abs(w)
winning_value_f = abs(w)
L_bidding_values = abs(eligible_bidding_value)
L_bidding_demand = abs(eligible_bidding_demand)
println("L_bidding_values",L_bidding_values)
println("L_bidding_demand", L_bidding_demand)

new_winning_values=zeros(0)
###############################################################################
# Price determination
i=1
while i <= length(winning_value) && length(L_bidding_values) > 0
	winner_remove = winning_value[i]
	j =findfirst(L_bidding_values, winner_remove)
	L_bidding_values=deleteat!(L_bidding_values,j)
	L_bidding_demand=deleteat!(L_bidding_demand,j)
	new_max_bid_value = maximum(L_bidding_values)  # Find maximum bidding values
	new_winning_values = append!(new_winning_values, new_max_bid_value)
 i += 1
 end
# The chosen outcome when each winning MVNO i is not participating
println("New winning values  when each winning MVNO i is not participating", new_winning_values)

l=1
j=length(new_winning_values)
# Choosen outcome, when each winning MVNO i is not participating
while l <= j
	winnervalue = winning_value[l]
	other_winning_value = new_winning_values[j]
	others_winners_no_m = winning_value[winning_value .≠ winnervalue]
	others_winners_no_m = append!(others_winners_no_m, other_winning_value)
	println(others_winners_no_m)
	others_winners_than_m=sum(others_winners_no_m)
	v_no_m = append!(v_no_m,others_winners_than_m)
 l += 1
 end
	# The welfare of other players than MVNO m from the chosen outcome
	# when MVNO m is not participating in auction
println("Welfare of other players than MVNO m, when m is not there", v_no_m)

println("winning_value_f", winning_value_f)
i=1
# The chosen outcome when each winning MVNO m is participating in auction
while i <= length(winning_value_f)
	winner_value = winning_value_f[i]
	others_winners_no_m = winning_value_f[winning_value_f .≠ winner_value]
	println(others_winners_no_m )
	others_winner2=sum(others_winners_no_m)
	v_m = append!(v_m, 	others_winner2)
	i += 1
 end
	# The welfare of other players than MVNO m from the chosen outcome
	# when MVNO m participates in auction
println("Welfare of other players than MVNO m, when m is there", v_m)
social_optimal_price=v_no_m .-v_m
println("Social optimal price", social_optimal_price)

i = 1
while i <= length(w)
	k = findfirst(eligible_bidding_value, w[i])
	x_decision[k]=1  # update decision variables based on the winner
	P_optimal[k]=social_optimal_price[i]
	i += 1
end

# Allocate cache resources to the winners
println("x_decision", x_decision)
resource_allocation=eligible_bidding_demand .* x_decision
println("Cache resource allocation",resource_allocation)

# Payments based on demands for cache resources

payments= resource_allocation .* P_optimal
println("payments to InP for cache resources", payments)
end_time= Dates.now()
running_time = end_time - starting_time
println("Running Time:", running_time )
