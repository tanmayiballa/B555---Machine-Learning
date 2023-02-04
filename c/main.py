
input_file = './dice_100x3.in'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with open(input_file) as file:
        n,d = [int(x) for x in next(file).split()]
        index = n-2
        die = d
        dice = []
        while(index>=0):
            all_dice = []
            d_c = []
            while die!=0:
                d_c.append([int(x) for x in next(file).split()])
                all_dice.append(d_c)
                die-=1
            die = d
            dice.append(d_c)
            index-=1
            k = next(file)
        #print(dice_config)
        #print(n,d)

    vV = []
    vC = []
    choose = []
    dp = []

    for i in range(0,n):
        vV.append([])
        vC.append([])
        choose.append(-1)
        dp.append(-1)

    choose[n-1] = -1
    choose[n-2] = 0
    vV[n-2].append(1.0)
    vC[n-2].append(1)
    vV[n-1].append(1.0)
    vC[n-1].append(0)

    i = n-3
    #print(dice[0][0])
    while(i>=0):
        print("iteration ", i)
        min_val = 10000005.0
        for j in range(0,d):
            min_tmp = 0.0
            cnt_p = 0
            for p in range(0,6):
                i = int(i)
                if(dice[i][j][p] == 1):
                    cnt_p+=1
            for k in range(0,6):
                if(dice[i][j][k]==0):
                    continue
                nxt_sq = i+k+1
                for ele in range(0,len(vV[nxt_sq])):
                    min_tmp= min_tmp + (1.0/cnt_p)*vV[nxt_sq][ele]*(vC[nxt_sq][ele]+1)
                    min_tmp = float(min_tmp)

            if(min_tmp<min_val):
                min_val = float(min_tmp)
                choose[i] = j
                dp[i] = float(min_tmp)

        die = choose[i]
        cnt_p = 0
        for p in range(0, 6):
            if (dice[i][die][p] == 1):
                cnt_p += 1
        for k in range(0,6):
            if(dice[i][die][k] == 0):
                continue
            nS = i + k + 1;
            for ele in range(0, len(vV[nS])):
                vV[i].append(float((1.0 / cnt_p) * vV[nS][ele]))
                vC[i].append(int(vC[nS][ele] + 1))
        i-=1
    print(dp)
    print(choose)








