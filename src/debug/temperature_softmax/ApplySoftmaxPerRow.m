function y = ApplySoftmaxPerRow(z,T)

[r,c] = size(z);

y = zeros(r,c);

for i=1:r
    num = exp(z(i,:)/T);
    den = sum(num);
    y(i,:) = num/den;
end