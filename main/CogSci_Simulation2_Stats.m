disp(' ============== OPTIMIZED MODEL=============');

X = frequency_log(:);
Y = optimal_gain(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: optimal_gain ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);

X = frequency_log(:);
Y = overallRT_global(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: overallRT_global ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = overallER_global(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: overallER_global ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = incongruencyCostRT_global(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: incongruencyCostRT_global ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = incongruencyCostER_global(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: incongruencyCostER_global ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = switchCostRT_global(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: switchCostRT_global ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = switchCostER_global(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: switchCostER_global ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = overallRT_local(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: overallRT_local ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = overallER_local(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: overallER_local ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = incongruencyCostRT_local(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: incongruencyCostRT_local ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = incongruencyCostER_local(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: incongruencyCostER_local ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = switchCostRT_local(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: switchCostRT_local ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = switchCostER_local(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: switchCostER_local ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = relActivation_global(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: relActivation_global ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = irrelActivation_global(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: irrelActivation_global ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                


X = frequency_log(:);
Y = relActivation_local(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: relActivation_local ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = irrelActivation_local(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: irrelActivation_local ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

disp(' ============== UNOPTIMIZED MODEL=============');     

X = frequency_log(:);
Y = overallRT_global_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: overallRT_global_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = overallER_global_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: overallER_global_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = incongruencyCostRT_global_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: incongruencyCostRT_global_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = incongruencyCostER_global_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: incongruencyCostER_global_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = switchCostRT_global_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: switchCostRT_global_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = switchCostER_global_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: switchCostER_global_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = overallRT_local_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: overallRT_local_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = overallER_local_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: overallER_local_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = incongruencyCostRT_local_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: incongruencyCostRT_local_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = incongruencyCostER_local_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: incongruencyCostER_local_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = switchCostRT_local_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: switchCostRT_local_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = switchCostER_local_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: switchCostER_local_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = relActivation_global_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: relActivation_global_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = irrelActivation_global_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: irrelActivation_global_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                


X = frequency_log(:);
Y = relActivation_local_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: relActivation_local_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = frequency_log(:);
Y = irrelActivation_local_unoptimized(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: irrelActivation_local_unoptimized ~ frequency, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                
