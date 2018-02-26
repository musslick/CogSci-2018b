X = gain_log(:);
Y = overallPerformanceRT_g(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: overallPerformanceRT_g ~ gain, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);
                                                                                                
X = gain_log(:);
Y = switchCostsRT_g(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: switchCostsRT_g ~ gain, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                

X = gain_log(:);
Y = incongruencyCostsRT_g(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: incongruencyCostsRT_g ~ gain, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                
                                                                     
                                                                                                
X = gain_log(:);
Y = overallPerformanceER_g(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: overallPerformanceER_g ~ gain, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                
                                                                     
X = gain_log(:);
Y = switchCostsER_g(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: switchCostsER_g ~ gain, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                
                                                                     

X = gain_log(:);
Y = incongruencyCostsER_g(:);
lm = fitlm(X,Y,'linear');
an = anova(lm,'summary');
disp(['regression: incongruencyCostsER_g ~ gain, b = ' num2str(lm.Coefficients.Estimate(2))...
                                                                                                    ', t(' num2str(an{1, 2}) ...
                                                                                                    ') = ' num2str(lm.Coefficients.tStat(2)) ...
                                                                                                    ', p = ' num2str(lm.Coefficients.pValue(2))]);                                                                                                
                                                                     

