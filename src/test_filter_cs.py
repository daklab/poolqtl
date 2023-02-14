cs = "10M30N11M"
L = int(cs.split("N")[0].split("M")[-1])
cs_split_M=cs.split("M")
edge5, edge3 = cs_split_M[0], cs_split_M[-2].split("N")[-1]
minedge = min([int(edge5), int(edge3)])