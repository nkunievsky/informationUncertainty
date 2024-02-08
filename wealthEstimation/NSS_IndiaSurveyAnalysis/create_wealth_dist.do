clear all 
cd "D:\Dropbox\Dropbox\uchicago_fourth\uncertaintyInequality"
global datafolder = "proc_data"

use "${datafolder}\NSSAIDIS_2013_proc.dta", clear 


*Very helpful for the weights 
* https://www.emmanuelteitelbaum.com/post/working-with-nss-data-in-stata/

*Some renaming 
	ren State state
	lab var state "state code"
	ren Sector sector
	lab var sector "rural or urban"
	ren Stratum stratum
	lab var stratum "stratum"
	ren SubStratumNo substratum
	lab var substratum "substratum"
	ren Vill_Blk_Slno psu
	lab var psu "primary survey unit (village/block)"
	ren HG_SubBlkNo hamlet_subblock
	lab var hamlet_subblock "hamlet group or sub-block number"
	ren Second_Stratum ss_strata_no
	lab var ss_strata_no "second stage stratum number"
	ren b4q8 upas_code
	lab var upas_code "Usual principal activity status code"
	destring upas_code, replace

*Create first strata variable 
	gen fs_strata = state + sector + stratum + substratum
	lab var fs_strata "first stage strata"

*Create weights 
	g Multiplier_comb = MLT/100 if NSS == NSC 
	replace Multiplier_comb = MLT/200 if NSS != NSC 

*Define survey 
	svyset psu [pw = Multiplier_comb], strata(fs_strata) singleunit(centered)
	svydescribe

*Define total net assets 
foreach v in b5_2_6  b5_1_6  b6_q6  b7_q5  b8_q5 b9_q4 b11_q6 b12_q3 b13_q4 b14_q17 b15_q5{
	replace `v' = 0 if `v'==.
}

g totalAssets = b5_2_6 + b5_1_6  + b6_q6  + b7_q5  + b8_q5 + b9_q4 + b11_q6 + b12_q3 + b13_q4 - b14_q17 - b15_q5

sum totalAssets [aweight = Multiplier_comb], d
preserve
keep  totalAssets
replace totalAssets = totalAssets/1000
outreg2 totalAssets using "D:\Dropbox\Dropbox\uchicago_fourth\uncertaintyInequality\presentation_outputs\figures_slides_11062022\NSS_WealthDistribution.tex", sum(detail) eqdrop(N sum_w  Var skewness kurtosis sum) replace 
restore 


_pctile totalAssets [pweight=Multiplier_comb], p(1(4)100)
loc j = 1 
forv i = 1(4)100{
	disp `i' 
	disp `r(r`j')'
	loc j = `j'+1
	
}

g totalAssets_inflated= b5_2_6 + b5_1_6  + b6_q6  + b7_q5  + b8_q5 + b9_q4 + b11_q6 + b12_q3 + b13_q4 - (b14_q17 + b15_q5)*2
sum totalAssets_inflated[aweight = Multiplier_comb], d

_pctile totalAssets [pweight=Multiplier_comb], p(1(4)100)
loc j = 1 
forv i = 1(4)100{
	disp `i' 
	disp `r(r`j')'
	loc j = `j'+1
	
}

g totalAssets_noland= b7_q5  + b8_q5 + b9_q4 + b11_q6 + b12_q3 + b13_q4 - b14_q17 - b15_q5
sum totalAssets_noland[aweight = Multiplier_comb], d

_pctile totalAssets_noland[pweight=Multiplier_comb], p(1(4)100)
loc j = 1 
forv i = 1(4)100{
	disp `i' 
	disp `r(r`j')'
	loc j = `j'+1
	
}
