function y = ApplyBlockSoftmaxPerRow(z, blocks, T)

[r,c] = size(z);

y = zeros(r,c);

nTasks = size(blocks, 1);

for i=1:r
    % active_indices = find(mask(i,:) == 1);
    % y(i, active_indices) = ApplySoftmaxPerRow( z(i, active_indices), T);
    for j = 1:nTasks
        start_node = blocks(j,1); 
        end_node = blocks(j,2);
        active_nodes = start_node:end_node;
        if j == 1 % Temperature only for first softmax
          y(i, active_nodes) = ApplySoftmaxPerRow( z(i, active_nodes), T); 
        else 
          y(i, active_nodes) = ApplySoftmaxPerRow( z(i, active_nodes), 1);
        end
    end
        
end