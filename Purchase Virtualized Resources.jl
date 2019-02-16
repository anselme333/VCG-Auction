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
using convex
################################################################################
# Inputs
starting_time= Dates.now()
RANDOM_SEED = 1000
srand(RANDOM_SEED)
num_MVNO = 50


#while num_MVNO <= max_num_MVNO

# The vector of  bidding values per unit of cache storage (1 GB)
bid_MVNO = rand(Uniform(3.31,8.31),num_MVNO)
# The vector of cache demands from MVNOs in terms of Gigabyte
MVNO_Cache_Req = rand(Uniform(100,500), num_MVNO)
# Total cache capacity of infrastructure provider in terms of GB
InP_max_cache_capacity =2000

# Reserved price per unit of cache storage(1 GB) per day
# This price is the standard price from microsot azure
# They charge standard price of 0.138 usd per one hour
# https://azure.microsoft.com/en-us/pricing/details/cache/

reserved_cache_price = 3.31


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
total_payment=sum(payments)
sum_resource_allocation=sum(resource_allocation)

##################################################
#When MVNOs do not buy resource at BS
# backhaul_bandwidth=0.21608  per Mbps for 2018
backhaul_bandwidth = 0.21608
# MVNO has to pay transit bandwidth plus resource at DC
resource_dc = resource_allocation .* backhaul_bandwidth
resource_dc=
Profits = resource_dc
println("Profits", resource_dc)


fig1 = figure("pyplot_multiaxis1",figsize=(10,12))
bidder1=1:length(v_no_m);
plot(bidder1,v_no_m, marker="x", linewidth=2.0, markersize=20, label="Social welfare without each winner")
legend(fancybox="true", fontsize=20)

bidder2=1:length(v_m);
plot(bidder2,v_m, marker="o", markersize=20, linewidth=2.0, label="Social welfare with each winner")
legend(fancybox="true", fontsize=20)

price2=1:length(social_optimal_price);
plot(price2,social_optimal_price, marker="*", markersize=20, linewidth=2.0, label="Optimal price " )
legend(fancybox="true", fontsize=20)

ax1 = gca()
grid("on")
xlabel("Number of MVNOs", fontsize=20)
font1 = Dict("color"=>"blue")
ylabel(" Price (USD), Social welfare (USD) ",fontdict=font1, fontsize=20)
setp(ax1[:get_yticklabels](),color="blue", fontsize=20) # Y Axis font formatting
ax1[:tick_params]("both",labelsize=24)
fig1[:canvas][:draw]() # Update the figure
#savefig("C:/Users/anselme/Dropbox/project/social_warefare_price.pdf",dpi=95)
gcf() # Needed for IJulia to plot inline



fig2 = figure("pyplot_multiaxis2",figsize=(10,12))
resource_allocation=filter(x -> x > 0, resource_allocation) # Non zero allocation
resource=1:length(resource_allocation);
plot(resource,resource_allocation, marker="*", markersize=20, linewidth=2.0, label="Cache allocation of each MVNO (GB)" )
legend(fancybox="true", fontsize=20)
payments=filter(x -> x > 0, payments) # Non zero payments
pay=1:length(payments);
plot(pay,payments, marker="o", markersize=20, linewidth=2.0, label="Payment of each MVNO (USD)" )
legend(fancybox="true", fontsize=20)
Profits = filter(x -> x > 0, Profits) # Non zero allocation
profits_bidders=1:length(Profits);
plot(profits_bidders,Profits, marker="x", markersize=20, linewidth=2.0, label="Profit of each MVNO (USD)" )
legend(fancybox="true", fontsize=20)

ax2 = gca()
grid("on")
xlabel("Number of MVNOs", fontsize=20)
font1 = Dict("color"=>"blue")
#ylabel("Cache allocation (GB), Payment (USD)",fontdict=font1, fontsize=20)
setp(ax2[:get_yticklabels](),color="blue", fontsize=20) # Y Axis font formatting
ax2[:tick_params]("both",labelsize=24)
ax2[:ticklabel_format](style="sci",axis="y",scilimits=(0,0))
fig2[:canvas][:draw]() # Update the figure
#savefig("C:/Users/anselme/Dropbox/project/Payment_MVNO.pdf",dpi=95)
gcf() # Needed for IJulia to plot inline



end_time= Dates.now()
running_time = end_time - starting_time
println("Running Time:", running_time)
