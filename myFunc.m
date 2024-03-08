function myFunc(pd, pg)
    %import cal_grid_dcopf_v2
    mpc = CaliforniaTestSystem();
    mpc.bus(:,3) = pd;
    mpc.gen(:,2) = pg;
end