function f = removeBus(indices, idx_line)
	m = loadcase('CaliforniaTestSystem.m');
	indices = cell2mat(indices);
	idx_line = cell2mat(idx_line);
	%remove pmax = zero
	%idx = find(m.gen(:,9)==0);
	%m.gen(idx,:) = [];
	%m.gencost(idx, :) = [];
	%remove solar
	m.bus(indices,3) = 0;
	m.branch(idx_line,:) = [];
	f = savecase('CaliforniaTestSystem.m', m);
	%a = sum(m.bus(:,3));
	%f = fprintf('Power demand: %f/n', a);
end