function f = removeSolar(indices)
	m = loadcase('CaliforniaTestSystem.m');
	indices = cell2mat(indices);
	%remove pmax = zero
	%idx = find(m.gen(:,9)==0);
	%m.gen(idx,:) = [];
	%m.gencost(idx, :) = [];
	%remove solar
	for i = indices
		m.gen(i,9) = 0;
		m.gen(i,10) = 0;
		m.gen(i,2) = 0;
	end
	%for i = indices
		%m.gencost(i, 6) = 0;
		%m.gencost(i, 4) = 0;
	%end
	f = savecase('CaliforniaTestSystem.m', m);
	%a = sum(m.bus(:,3));
	%f = fprintf('Power demand: %f/n', a);
end