import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, sympify
import matplotlib.pyplot as plt
import math
from itertools import product
#from myhdl import block, always_comb, Signal, intbv


def detect_metastability(bitsets):
    # Determine the length of the bitsets
    bit_length = len(bitsets[0])
    
    # Initialize the result with zeros
    result = ['0'] * bit_length
    
    # Check each bit position for metastability
    for i in range(bit_length):
        # Extract the ith bit from each bitset
        bits_at_i = [bitset[i] for bitset in bitsets]
        
        # If not all bits are the same, mark as metastable
        if len(set(bits_at_i)) > 1:
            result[i] = 'M'
        else:
            # Otherwise, use the bit value (all the same)
            result[i] = bits_at_i[0]
    
    return ''.join(result)

# # Given bitsets
# bitsets = ["1","1","1","1"]

# # Detect metastability
# metastability = detect_metastability(bitsets)
# print(metastability)


def generate_combinations_list(bitset):
    # Replace 'M' in bitset with both possibilities '0' and '1'
    combinations = []
    if 'M' in bitset:
        # Generate combinations for 'M' by replacing it with '0' and '1'
        for bit in ['0', '1']:
            combinations += generate_combinations_list(bitset.replace('M', bit, 1))
    else:
        # No 'M' found, add the bitset as is
        combinations.append(bitset)
    
    return combinations

# # Example bitset with metastability
# bitset_example = "MM0"
# x=generate_combinations_list(bitset_example)
# print(x)

def generate_combinations_indexes(bitset, indices):
    # Convert bitset to a list to modify specific indices
    bitset_list = list(bitset)
    combinations = []

    # Recursive helper function to generate combinations
    def generate_helper(current_bitset, idx):
        if idx == len(indices):
            combinations.append(''.join(current_bitset))
            return
        # Get the current index to modify
        current_index = indices[idx]
        # Replace at the current index with '0' and '1', then recurse
        for bit in ['0', '1']:
            current_bitset[current_index] = bit
            generate_helper(current_bitset, idx + 1)

    # Start the recursive generation
    generate_helper(bitset_list, 0)
    return combinations

# # Example usage
# bitset_example = "M10"
# indices = [0, 2]
# result = generate_combinations_indexes(bitset_example, indices)
# print(result)







def plot_bitset_expression_result(bitset, expression):
    # Define symbols for the expression (up to 26 variables)
    variables = symbols('a:z')
    # Map bitset to variable values
    values = {variables[i]: bool(int(bitset[i])) for i in range(len(bitset))}
    
    # Evaluate the expression
    expr = sympify(expression)
    result = expr.subs(values)

    t=0

    if (result):
        t=1
    
    return(t)

# #Example usage
# bitset = "10"
# expression = "~a&~b"
# y=plot_bitset_expression_result(bitset, expression)
# print(y)


def integrated_metastability_detection(bitset_with_m, expression):
    # Generate all combinations from the bitset with 'M'.
    bitsets = generate_combinations_list(bitset_with_m)
    
    # Evaluate each bitset against the expression and collect results.
    results = [str(plot_bitset_expression_result(bitset, expression)) for bitset in bitsets]
    
    # Analyze the results for metastability.
    metastability_result = detect_metastability(results)
    
    # Return the metastability detection result.
    return metastability_result

# # Example usage.
# bitset_with_m = "01M"
# expression = "a&~c|b&c|a&b"
# result = integrated_metastability_detection(bitset_with_m, expression)
# print(result)  # This will print "0", "1", or "M" based on the metastability analysis.

def integrated_metastability_detection_comments(bitset_with_m, expression):
    # Generate all combinations from the bitset with 'M'.
    bitsets = generate_combinations_list(bitset_with_m)
    print("The all posiblle combinations are:\n",bitsets)
    
    # Evaluate each bitset against the expression and collect results.
    results = [str(plot_bitset_expression_result(bitset, expression)) for bitset in bitsets]
    print("The all resullts by passing those combinations in teh exprassio are:\n",results)

    # Analyze the results for metastability.
    metastability_result = detect_metastability(results)
    
    # prints the metastability detection result.
    print("The final resullt of the * operation between the resullts is:\n",metastability_result)
    return metastability_result

    


# def eval_bit_operation(a, b, op):
#     """Evaluate bit operations with support for a metastable 'M'."""
#     if op == '&':
#         if a == '0' or b == '0':
#             return '0'
#         elif a == 'M' or b == 'M':
#             return 'M' if a == '1' or b == '1' else '0'
#         else:
#             return '1'
#     elif op == '|':
#         if a == '1' or b == '1':
#             return '1'
#         elif a == 'M' or b == 'M':
#             return 'M' if a == '0' or b == '0' else '1'
#         else:
#             return '0'

# def preprocess_expression(expression, bit_values):
#     """Preprocess the expression to replace variables with their bit values."""
#     for var, val in bit_values.items():
#         expression = expression.replace(var, val)
#     return expression

# def evaluate_expression_with_M(bitset, expression):
#     # Assuming variables in expressions are 'a', 'b', 'c', etc.
#     variables = 'abcdefghijklmnopqrstuvwxyz'
#     bit_values = {variables[i]: bitset[i] for i in range(len(bitset))}
    
#     # Preprocess the expression to replace variables with bit values
#     preprocessed_expr = preprocess_expression(expression, bit_values)

#     # Example simplification assuming binary operations only
#     # This needs to be expanded for more complex expressions
#     # The example below handles expressions like 'a&b' only
#     parts = preprocessed_expr.split('&') if '&' in preprocessed_expr else preprocessed_expr.split('|')
#     op = '&' if '&' in preprocessed_expr else '|'

#     result = eval_bit_operation(parts[0], parts[1], op)
#     return result

# # Example usage
# bitset = 'MM'
# expression = 'a|b'
# result = evaluate_expression_with_M(bitset, expression)
# print(result)  





from itertools import combinations

def generate_n_choose_k(n, k):
    # Generate all combinations of n items taken k at a time without repetition
    items = range(n)  # Items represented by indices starting from 0 to n-1
    all_combinations = list(combinations(items, k))
    # Convert tuples to lists for the output format you requested
    return [list(comb) for comb in all_combinations]

# # Example usage
# n = 3
# k = 2
# combinations_list = generate_n_choose_k(n, k)
# print(combinations_list)

def compute_all_unions(sets_list):
    all_unions = []
    
    # Iterate over each pair of sets (including each set with itself)
    for i in sets_list:
        for j in sets_list:
            # Compute the union and add it to the result list
            union_set = sorted(list(set(i).union(set(j))))
            all_unions.append(union_set)
    
    return all_unions

# # Example usage
# sets_list = [[0, 1], [0, 2], [1, 2]]
# all_unions = compute_all_unions(sets_list)
# print(all_unions)


def mask_bitset_with_m(bitset, indices):
    # Convert the bitset string to a list of characters for easier manipulation
    bitset_list = list(bitset)
    
    # Iterate through the indices list and replace the corresponding characters with 'M'
    for index in indices:
        if 0 <= index < len(bitset_list):  # Check if the index is within the bounds of the bitset
            bitset_list[index] = 'M'
    
    # Convert the list back to a string
    masked_bitset = ''.join(bitset_list)
    
    return masked_bitset

# # Example usage
# bitset = "00110101010101"
# indices = [0, 5]
# masked_bitset = mask_bitset_with_m(bitset, indices)
# print(masked_bitset)


def cmux2_1(a,b,s):
     bitset=str(a)+str(b)
     used_bitset=bitset+str(s) 
     results=integrated_metastability_detection(used_bitset,"a&~c|b&c|a&b")
    #  used_bitset1=bitset+str(1) 
    #  results.append(integrated_metastability_detection(used_bitset1,"a&~c|b&c|a&b"))


     return results


# #example
# a=1
# b='M'
# s=0
# TT=cmux2_1(a,b,s)
# print(TT)

# The bitset counts from left to right, i.g:bitset= 0100 with s=01 will return 1.
def cmux_n_1(bitset,s):
    if len(bitset)==2:
        a=bitset[0]
        b=bitset[1]
        return cmux2_1(a,b,s)
    else:
        midpoint=len(bitset)//2
        bitset1=bitset[:midpoint]
        bitset2=bitset[midpoint:]
        snew=s[1:]
        a=cmux_n_1(bitset1,snew)
        b=cmux_n_1(bitset2,snew)
        return cmux2_1(a,b,s[0])


# bitset="0000000000000000"
# s="1110"
# print(cmux_n_1(bitset,s))


def and_tree(bitset):
    if len(bitset)==2:
        return integrated_metastability_detection(bitset,"a&b")
    else:
        midpoint=len(bitset)//2
        bitset1=bitset[:midpoint]
        bitset2=bitset[midpoint:]
        newbitset=[]
        newbitset.append(and_tree(bitset1))
        newbitset.append(and_tree(bitset2))
        strnewbitset=''.join(str(num) for num in newbitset)
        return integrated_metastability_detection(strnewbitset,"a&b")


# #example:
# bitset="101M"   
# print(and_tree(bitset))


    
def or_tree(bitset):
    if len(bitset)==2:
        return integrated_metastability_detection(bitset,"a|b")
    else:
        midpoint=len(bitset)//2
        bitset1=bitset[:midpoint]
        bitset2=bitset[midpoint:]
        newbitset=[]
        newbitset.append(or_tree(bitset1))
        newbitset.append(or_tree(bitset2))
        strnewbitset=''.join(str(num) for num in newbitset)
        return integrated_metastability_detection(strnewbitset,"a|b")
    

# #example:
# bitset="0000000000M0000"   
# print(or_tree(bitset))

def extend_to_next_power_of_two_ones(bitset):
    # Determine the current length of the bitset
    current_length = len(bitset)
    
    # Find the next power of 2 greater than or equal to the length of the bitset
    if current_length & (current_length - 1) == 0:
        # Already a power of two
        next_power_of_two = current_length
    else:
        next_power_of_two = 2 ** math.ceil(math.log2(current_length))
    
    # Calculate the number of '1's to add
    number_of_ones_to_add = next_power_of_two - current_length
    
    # Extend the bitset with '1's
    extended_bitset = bitset + '1' * number_of_ones_to_add
    
    return extended_bitset

def extend_to_next_power_of_two_zeros(bitset):
    # Determine the current length of the bitset
    current_length = len(bitset)
    
    # Find the next power of 2 greater than or equal to the length of the bitset
    if current_length & (current_length - 1) == 0:
        # Already a power of two
        next_power_of_two = current_length
    else:
        next_power_of_two = 2 ** math.ceil(math.log2(current_length))
    
    # Calculate the number of '1's to add
    number_of_ones_to_add = next_power_of_two - current_length
    
    # Extend the bitset with '0's
    extended_bitset = bitset + '0' * number_of_ones_to_add
    
    return extended_bitset


def circute_comments(n,k,expr,bitset):
    combinations=generate_n_choose_k(n,k)
    print("The combinations of n choose k are:",combinations)
    unions=compute_all_unions(combinations)
    print("The all posible unions are: ", unions)
    m_unions=[]
    for union in unions:
        m_unions.append(mask_bitset_with_m(bitset,union))
    print("The all union posibillitys with m are:",m_unions)
    mux_res=[]
    list_to_mux=[]
    bits_to_mux=""
    s=""
    counter=0
    for union in m_unions:
        ###########
        print(f"mux number {counter+1}")
        list_to_mux=[]
        s=""
        bits_to_mux=""
        list_to_mux=generate_combinations_indexes(union,unions[counter])
        ############
        print("list to mux= ",list_to_mux)
        for set in list_to_mux:
            bits_to_mux+=str(integrated_metastability_detection(set,expr))
        ############
        print("bits to mux= ",bits_to_mux)
        for i in unions[counter]:
            s+=str(bitset[i])
        ############
        print("s= ",s)
        mux_res.append(cmux_n_1(bits_to_mux,s))

        counter+=1
 
    ###########
    print("mux res= ",mux_res)

    and_gate_size=len(combinations)
    bitsets_to_and=[]
    for i in range(0,len(mux_res),and_gate_size):
        bitsets_to_and.append(mux_res[i:i+and_gate_size])

    ##############
    print("bitsets_to_and=",bitsets_to_and)

    and_res=[]
    for set in bitsets_to_and:
        s=""
        for i in set:
            s+=str(i)
        s=extend_to_next_power_of_two_ones(s)
        and_res.append(and_tree(s))
    
    ##############
    print("and gates resullts: ",and_res)
    
    s=""
    for res in and_res:
        s+=str(res)

    s=extend_to_next_power_of_two_zeros(s)
    final_resullt=or_tree(s)

    ###############
    print(f"The final resuult is: {final_resullt}\n")

    return final_resullt

def circute(n,k,expr,bitset):
    combinations=generate_n_choose_k(n,k)
    unions=compute_all_unions(combinations)
    m_unions=[]
    for union in unions:
        m_unions.append(mask_bitset_with_m(bitset,union))
    mux_res=[]
    list_to_mux=[]
    bits_to_mux=""
    s=""
    counter=0
    for union in m_unions:
        
        list_to_mux=[]
        s=""
        bits_to_mux=""
        list_to_mux=generate_combinations_indexes(union,unions[counter])
        
        for set in list_to_mux:
            bits_to_mux+=str(integrated_metastability_detection(set,expr))
        
        for i in unions[counter]:
            s+=str(bitset[i])
        
        mux_res.append(cmux_n_1(bits_to_mux,s))

        counter+=1
    
    

    and_gate_size=len(combinations)
    bitsets_to_and=[]
    for i in range(0,len(mux_res),and_gate_size):
        bitsets_to_and.append(mux_res[i:i+and_gate_size])

    

    and_res=[]
    for set in bitsets_to_and:
        s=""
        for i in set:
            s+=str(i)
        s=extend_to_next_power_of_two_ones(s)
        and_res.append(and_tree(s))
    
    
    
    s=""
    for res in and_res:
        s+=str(res)

    s=extend_to_next_power_of_two_zeros(s)
    final_resullt=or_tree(s)

    

    return final_resullt

def closest_power_of_2(n):
    # If the number is already a power of two or is less than 1, return the number itself
    if n < 1:
        return 1
    power = 1
    while power < n:
        power <<= 1  # Shift left by one bit (equivalent to multiplying by 2)
    return power

# # Test the function
# input_number = 10
# result = closest_power_of_2(input_number)
# print(f"The closest power of 2 to {input_number} is {result}")

def gates(n,k):
    num_and=0
    num_or=0
    num_not=0
    combinations_num=math.comb(n, k)
    size_of_tree=closest_power_of_2(combinations_num)
    for i in range (0,int(math.log(size_of_tree,2))-1):
        num_and=num_and+2**i
        num_or=num_or+2**i
    num_and=num_and*combinations_num
    x=generate_n_choose_k(n,k)
    y=compute_all_unions(x)
    mux_2=0
    expressions=0
    for i in y:
        expressions=expressions+2**len(i)
        for j in range (0,len(i)-1):
            mux_2=mux_2+2**j
    
    num_or=num_or+mux_2*3
    num_and=num_and+mux_2*3
    num_not=mux_2

    print(f"Total:\nand gates: {num_and}\nor gates: {num_or}\nnot gates: {num_not}\nalso the number of gates that are nedded for the expression are multipled by {expressions}")

def gates_no_comments(n,k):
    num_and=0
    num_or=0
    num_not=0
    combinations_num=math.comb(n, k)
    size_of_tree=closest_power_of_2(combinations_num)
    for i in range (0,int(math.log(size_of_tree,2))-1):
        num_and=num_and+2**i
        num_or=num_or+2**i
    num_and=num_and*combinations_num
    x=generate_n_choose_k(n,k)
    y=compute_all_unions(x)
    mux_2=0
    expressions=0
    for i in y:
        expressions=expressions+2**len(i)
        for j in range (0,len(i)-1):
            mux_2=mux_2+2**j
    
    num_or=num_or+mux_2*3
    num_and=num_and+mux_2*3
    num_not=mux_2

    return num_and+num_not+num_or

   


def letancy(n,k):
    or_way=0
    and_way=0
    not_way=0
    mux_2_way=0
    combinations_num=math.comb(n, k)
    size_of_tree=closest_power_of_2(combinations_num)
    or_way=int(math.log(size_of_tree,2))
    and_way=int(math.log(size_of_tree,2))
    x=generate_n_choose_k(n,k)
    y=compute_all_unions(x)
    lenths=[len(i) for i in y ]
    mux_2_way=max(lenths)
    not_way=mux_2_way
    or_way=or_way+mux_2_way*3
    and_way=and_way+mux_2_way
    return and_way*50+or_way*50+not_way*20

def plot_graphs(n):
    k_values = np.arange(1, n + 1)
    letancy_values = [letancy(n, k) for k in k_values]
    gates_values = [gates_no_comments(n, k) for k in k_values]

    # Plotting the letancy graph
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, letancy_values, marker='o')
    plt.title(f'Letancy, n={n}')
    plt.xlabel('k')
    plt.ylabel('PicoSeconds')
    
    # Plotting the gates graph
    plt.subplot(1, 2, 2)
    plt.plot(k_values, gates_values, marker='x', color='red')
    plt.title(f'Gates, n={n}')
    plt.xlabel('k')
    plt.ylabel('Gates')
    
    plt.tight_layout()
    plt.show()

def plot_heatmaps(n, max_k):
    n_values = np.arange(1, n + 1)
    k_values = np.arange(1, max_k + 1)
    letancy_matrix = np.zeros((max_k, n))
    gates_matrix = np.zeros((max_k, n))
    
    for i, k in enumerate(k_values):
        for j, n_val in enumerate(n_values):
            if k > n_val:
                letancy_matrix[i, j] = np.nan  # Mark as NaN for special color handling
                gates_matrix[i, j] = np.nan
            else:
                letancy_matrix[i, j] = letancy(n_val, k)
                gates_matrix[i, j] = gates_no_comments(n_val, k)

    # Define the colormap and color for NaN values
    cmap = plt.cm.viridis
    cmap.set_bad(color='black')

    # Plotting the letancy heatmap
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(letancy_matrix, extent=[1, n, 1, max_k], aspect='auto', cmap=cmap)
    plt.colorbar(label='PicoSeconds')
    plt.title(f'Letancy Heatmap, Max k={max_k}')
    plt.xlabel('n')
    plt.ylabel('k')
    
    # Plotting the gates heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(gates_matrix, extent=[1, n, 1, max_k], aspect='auto', cmap=cmap)
    plt.colorbar(label='Gates')
    plt.title(f'Gates Heatmap, Max k={max_k}')
    plt.xlabel('n')
    plt.ylabel('k')
    
    plt.tight_layout()
    plt.show()

    






# n=int(input("n=\n"))
# k=int(input("k=\n"))
# bitset=input("bitset=\n")
# expr=input("expression=\n")
# circute(n,k,expr,bitset)

# n=5
# k=2
# bitset="0MMMM"
# expr="a&b&c&d&e"
# circute(n,k,expr,bitset)

n=int(input("Enter the size of the bit set\n"))
k=int(input("Enter the size of the metastabillic bits that should come in the input\n"))
option=int(input("Please press 1 if you want to see a resullt for a spesific bitset \nPress 2 if you \
want to see the resullts for an every combination of \
bitset that can be in the size that you have choosen\nPress 3 for checking letancy and number of gates on range of 1-k in n\
\nPress 4 for checking the all range betwwn 1 to k and 1 to n when k<=n\n"))
if (option==1):
    expr=input("please enter a boolean expression, use \"&\" for \"and\",\"|\" for \"or\",\"~\" for \"not\"\nFor the expression use the bit from 'a' to 'z for example \"(~a&b)|c\n")
    bitset=input("Please enter the spesific bitset you want to check it on, for example: 'M10'\n")
    print("Using the Fm(x) *res(x):")
    x=integrated_metastability_detection_comments(bitset,expr)
    print("Now it will demontrate the Cmux circute which find the resullt by hardware\na good resullt only is garuntid if the numbers of M in the bitset is smaleer than k\n")
    y=circute_comments(n,k,expr,bitset)
    print(f"As we can see the resullt from the circute is {y} when the resullt from the res function is {x} in this case they are {'equal' if x == y else 'different'}")
    gates(n,k)
    print(f"If we guess that and gate letancy= 50 PicoSeconds, or gate letancy= 50 PicoSeconds,\nnot gate latency= 20 Pico Seconds")
    print(f"So the total circute latency is {letancy(n,k)} PicoSeconds + the latency of the expressions boolean circute")
elif (option==2):
    expr=input("please enter a boolean expression, use \"&\" for \"and\",\"|\" for \"or\",\"~\" for \"not\"\nFor the expression use the bit from 'a' to 'z for example \"(~a&b)|c\n")
    # Generate all possible bitsets of length n using '0', '1', and 'M'
    bitsets = [''.join(bits) for bits in product('01M', repeat=n)]

    same_count = 0
    different_count = 0

    for bitset in bitsets:
        m_count = bitset.count('M')
        comparison = "smaller" if m_count < k else "equal" if m_count == k else "bigger"
        print(f"The number of the M bits is {comparison} than k={k}")

        #function logic for integrated_metastability_detection
        result1 =integrated_metastability_detection(bitset,expr)

        #function logic for circuit
        result2 =  circute(n,k,expr,bitset)  

        print(f"{bitset} -> Result from res function: {result1}")
        print(f"{bitset} -> Result from Cmux circute: {result2}")

        if result1 == result2:
            print("True")
            same_count += 1
        else:
            print("False")
            different_count += 1

    print(f"Same results count: {same_count}")
    print(f"Different results count: {different_count}")
    gates(n,k)
    print(f"If we guess that and gate letancy= 50 PicoSeconds, or gate letancy= 50 PicoSeconds,\nnot gate latency= 20 Pico Seconds")
    print(f"So the total circute latency is {letancy(n,k)} PicoSeconds + the latency of the expressions boolean circute")

elif(option==3):
    plot_graphs(n)

elif(option==4):
    plot_heatmaps(n,k)












    
    












