import json

if __name__ == '__main__':
    with open("flops/r50.json", "r") as f:
        flops = json.load(f)

    with open("flops/r50-frs.json", "r") as f:
        frs = json.load(f)

    sops = 0
    power = 0
    for k, v in flops.items():
        if k in frs.keys():
            for k1, v1 in v.items():
                sops += v1 * frs[k][k1]*4
                power += (0.9* v1 * frs[k][k1]*4)
        else:
            sops += v
            power += 4.6*v

    print(sops/(10**9), power/(10**9))