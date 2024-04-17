function f = myFunc(real_geni)
	real_geni = double(real_geni);
	m = loadcase('CaliforniaTestSystem.m');
	m.bus(:,3) = (real_geni/sum(m.bus(:,3)))*m.bus(:,3);
	%U = unique(m.gen(:,2));
	m.gen(:,2) = (real_geni/sum(m.gen(:,2)))*m.gen(:,2);
	m.bus(:,4) = (real_geni/sum(m.bus(:,4)))*m.bus(:,4);
	f = savecase('CaliforniaTestSystem.m', m);
	%a = sum(m.bus(:,3));
	%f = fprintf('Power demand: %f/n', a);
end