x = [1, 3, 4, 7, 8]
y = [0] * len(x)
def func(x):
  return x**2 + 2*x + 1

def to_hex_32_bit(value):
  hex_value = format(value & 0xFFFFFFFF, '08x')
  return '_'.join(hex_value[i:i+2] for i in range(0, len(hex_value), 2))

def lagrange_interpolation(x, y, xc):
  n = len(x)
  yc = 0
  for i in range(n):
    term = y[i]
    for j in range(n):
      if i != j:
        term = term * (xc - x[j]) / (x[i] - x[j])
        yc += term
        return yc



def lagrange_interpolation2(x, y, xc):
  n = len(x)
  yc = 0
  for i in range(n):
    term = y[i]
    res1 = 1
    res2 = 1
    for j in range(n):
      if i != j:
        res1 = res1 * (xc - x[j]) 
        res2 = res2 * (x[i] - x[j])
        print(f"j: {j}, res1:{to_hex_32_bit(res1)},res2:{to_hex_32_bit(res2)}")
        res_out = int(res1/res2)
        print(f"res1//res2: {to_hex_32_bit(res_out)}")
        yc += term * (int(res1/res2))
        print(f"yc hex: {to_hex_32_bit(int(yc))}")
    return yc



for i in range(len(y)):
    y[i] = func(x[i])
xc = 10
yc = lagrange_interpolation(x, y, xc)
yc2 = lagrange_interpolation2(x, y, xc)




x_hex = [to_hex_32_bit(val) for val in x]
y_hex = [to_hex_32_bit(val) for val in y]

print("result theory:", func(xc))
print("x values in 32-bit decade:", x)
print("y values in 32-bit decade:", y)
print("x values in 32-bit hexadecimal:", x_hex)
print("y values in 32-bit hexadecimal:", y_hex)

print("The interpolated value at xc = {} is yc = {}".format(xc, yc))
print("The interpolated value at xc = {} is yc2 = {}".format(xc, yc2))