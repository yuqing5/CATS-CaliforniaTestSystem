function f = myFunc(real_geni)
	m = loadcase('CaliforniaTestSystem.m');
	m.bus(:,3) = (real_geni/sum(m.bus(:,3)))*m.bus(:,3);
	U = unique(m.gen(:,2));
	m.gen(:,2) = (real_geni/sum(U))*m.gen(:,2);
	m.bus(:,4) = (real_geni/sum(m.bus(:,4)))*m.bus(:,4);
	savecase('CaliforniaTestSystem.m', m);
	f = sum(m.bus(:,3));
end