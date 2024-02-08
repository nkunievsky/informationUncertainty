****************
** 2013/2012 ***
****************
clear all 
cd "D:\Dropbox\Dropbox\uchicago_fourth\uncertaintyInequality"
global datafolder = "NSS_AIDIS\2013\Visit1\NSS-70-18pt2-visit1_new_format\NSS-70-18pt2-visit1_new_format"

/*
Questionare is called schdule_18.2v1
*/

*Blocks 1 and 2 
use "${datafolder}\blocks1and2.dta", clear 
	*Check 
		unique HHID 
*Blocks 3 
	merge 1:1 HHID using "${datafolder}/blocks3.dta"
	drop _merge 
*Blocks 4
	tempfile tempi
	preserve 
	use "${datafolder}/blocks4.dta", clear 
	destring b4q3 , replace  
	keep if b4q3 == 1 
	save `tempi'
	restore 
	merge 1:1 HHID using `tempi', nogen 
*Block 5 
	merge 1:1 HHID using "${datafolder}/blocks5.dta", nogen 

*Block 5pt1 
	tempfile tempi
	preserve 
	use "${datafolder}/blocks5pt1.dta", clear 
	keep if b5_1_1 == "99"
	label var  b5_1_6 "land owned - Rural (Rs)"
	save `tempi'
	restore 
	merge 1:1 HHID using `tempi', nogen 

*Block 5pt2
	tempfile tempi
	preserve 
	use "${datafolder}/blocks5pt2.dta", clear 
	keep if b5_2_1 == "99"
// 	collapse (sum) b5_2_6, by(HHID)
	label var  b5_2_6 "land owned - Urban (Rs)"
	save `tempi'
	restore 
	merge 1:1 HHID using `tempi', nogen 

*Block 6 
	tempfile tempi
	preserve 
	use "${datafolder}/blocks6.dta", clear 
	keep if b6_q3 =="11"
	label var b6_q6 "Land Owned - Construction (Rs)"
	save `tempi'
	restore 
	merge 1:1 HHID using `tempi', nogen 
	
*Blocks 7 
	tempfile tempi
	preserve 
	use "${datafolder}/blocks7.dta", clear 
	keep if b7_q2=="22" //This sums the entire table 
	save `tempi'
	restore 
	merge 1:1 HHID using `tempi', nogen 


*Blocks 8
	tempfile tempi
	preserve 
	use "${datafolder}/blocks8.dta", clear 
	keep if b8_q2=="08" //This sums the entire table 
	label var b8_q5 "Owned Value - Transportation (Rs)"
	save `tempi'
	restore 
	merge 1:1 HHID using `tempi', nogen 
	
*Blocks 9
	tempfile tempi
	preserve 
	use "${datafolder}/blocks9.dta", clear 
	keep if b9_q2=="08" //This sums the entire table 
	label var b9_q4 "Owned Value - agricultural machinery  (Rs)"
	save `tempi'
	restore 
	merge 1:1 HHID using `tempi', nogen 
	
	
*Blocks 10
	tempfile tempi
	preserve 
	use "${datafolder}/blocks10.dta", clear 
	keep if b10_q2=="15" //This sums the entire table 
	label var b10_q3 "Owned Value - non-farm business equipment (Rs)"
	save `tempi'
	restore 
	merge 1:1 HHID using `tempi', nogen 

	
*Blocks 11
	tempfile tempi
	preserve 
	use "${datafolder}/blocks11.dta", clear 
	keep if b11_q1=="05" //This sums the entire table 
	label var b11_q6 "Owned Value - shares & debentures(Rs)"
	save `tempi'
	restore 
	merge 1:1 HHID using `tempi', nogen 
	
*Blocks 12
	tempfile tempi
	preserve 
	use "${datafolder}/blocks12.dta", clear 
	keep if b12_q1=="11" //This sums the entire table 
	label var b12_q3 "Owned Value - financial assets other than shares and debentures (Rs)"
	save `tempi'
	restore 
	merge 1:1 HHID using `tempi', nogen 


*Blocks 13
	tempfile tempi
	preserve 
	use "${datafolder}/blocks13.dta", clear 
	keep if b13_q2=="07" //This sums the entire table 
	save `tempi'
	restore 
	merge 1:1 HHID using `tempi', nogen 

*Blocks 14
	tempfile tempi
	preserve 
	use "${datafolder}/blocks14.dta", clear 
	keep if b14_q1=="99" //This sums the entire table 
	keep if b14_q17 !=. //Keeps only results for loans that were taken up to midst of 2012 
	save `tempi'
	restore 
	merge 1:1 HHID using `tempi', nogen 
	

*Blocks 15 
	tempfile tempi
	preserve 
	use "${datafolder}/blocks15.dta", clear 
	keep if b15_q1=="99" //This sums the entire table 
	save `tempi'
	restore 
	merge 1:1 HHID using `tempi', nogen 

save "proc_data\NSSAIDIS_2013_proc.dta", replace 



****************
** 2019/2018 ***
****************
clear all 
cd "D:\Dropbox\Dropbox\uchicago_fourth\uncertaintyInequality"
global datafolder = "NSS_AIDIS\2019\Round77sch18pt2Data"

*Block 1 
use "${datafolder}\visit1_blocks1and2.dta", clear 
	*Check 
		unique HHID 
		
*Blocks 3
use "${datafolder}/visit1_blocks3.dta", clear 
	merge 1:1 HHID using "${datafolder}/visit1_blocks3.dta"
	drop _merge 
	
*Blocks 4
	tempfile tempi
	preserve 
	use "${datafolder}/blocks4.dta", clear 
	destring b4q3 , replace  
	keep if b4q3 == 1 
	save `tempi'
	restore 
	merge 1:1 HHID using `tempi', nogen 
*Block 5 
	merge 1:1 HHID using "${datafolder}/blocks5.dta", nogen 



