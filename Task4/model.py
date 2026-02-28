import pulp
import matplotlib.pyplot as plt

#DEFINE SETS
factories = ["F1", "F2"]
warehouses = ["W1", "W2", "W3"]
products = ["P1", "P2"]

#PARAMETERS
capacity = {
    "F1": 500,
    "F2": 600
}

# Demand for each product at each warehouse
demand = {
    ("W1","P1"): 200,
    ("W1","P2"): 150,
    ("W2","P1"): 180,
    ("W2","P2"): 120,
    ("W3","P1"): 150,
    ("W3","P2"): 100
}

# Production cost per unit
production_cost = {
    ("F1","P1"): 10,
    ("F1","P2"): 12,
    ("F2","P1"): 9,
    ("F2","P2"): 11
}

# Shipping cost per unit
shipping_cost = {
    ("F1","W1"): 4,
    ("F1","W2"): 6,
    ("F1","W3"): 8,
    ("F2","W1"): 5,
    ("F2","W2"): 4,
    ("F2","W3"): 6
}

#CREATE MODEL
model = pulp.LpProblem("Multi_Product_Supply_Chain", pulp.LpMinimize)

#DECISION VARIABLES
# x[f,w,p] = units of product p shipped from factory f to warehouse w

x = pulp.LpVariable.dicts(
    "Ship",
    [(f, w, p) for f in factories for w in warehouses for p in products],
    lowBound=0,
    cat="Continuous"
)

#OBJECTIVE FUNCTION
# Minimize total production + shipping cost

model += pulp.lpSum(
    (production_cost[(f,p)] + shipping_cost[(f,w)]) * x[(f,w,p)]
    for f in factories
    for w in warehouses
    for p in products
)

#CONSTRAINTS

# --- Factory capacity constraint ---
for f in factories:
    model += pulp.lpSum(
        x[(f,w,p)]
        for w in warehouses
        for p in products
    ) <= capacity[f]

# --- Warehouse demand constraint ---
for w in warehouses:
    for p in products:
        model += pulp.lpSum(
            x[(f,w,p)]
            for f in factories
        ) == demand[(w,p)]

#SOLVE MODEL

model.solve()

# PRINT RESULTS
print("===================================")
print("Optimization Status:", pulp.LpStatus[model.status])
print("===================================\n")

print("Optimal Distribution Plan:\n")

for f in factories:
    for w in warehouses:
        for p in products:
            value = x[(f,w,p)].varValue
            if value > 0:
                print(f"{f} -> {w} ({p}) : {value:.2f} units")

print("\n===================================")
print("Minimum Total Cost = $", pulp.value(model.objective))
print("===================================")

# VISUALIZATION

labels = []
values = []

for f in factories:
    for w in warehouses:
        for p in products:
            val = x[(f,w,p)].varValue
            if val > 0:
                labels.append(f"{f}-{w}-{p}")
                values.append(val)

plt.figure(figsize=(12,6))
plt.bar(labels, values)
plt.xticks(rotation=90)
plt.title("Optimal Shipment Quantities")
plt.xlabel("Factory-Warehouse-Product")
plt.ylabel("Units Shipped")
plt.tight_layout()
plt.show()
