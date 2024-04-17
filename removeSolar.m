function f = removeSolar(indices)
	m = loadcase('CaliforniaTestSystem.m');
	indices = cell2mat(indices);
	%remove pmax = zero
	%idx = find(m.gen(:,9)==0);
	%m.gen(idx,:) = [];
	%m.gencost(idx, :) = [];
	%remove solar
	m.gen(indices,:) = [];
	m.gencost(indices, :) = [];
	f = savecase('CaliforniaTestSystem.m', m);
	%a = sum(m.bus(:,3));
	%f = fprintf('Power demand: %f/n', a);
end