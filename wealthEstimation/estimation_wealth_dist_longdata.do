clear all 
set maxvar 30000

cd "C:\Users\nadav\Dropbox\uchicago_fourth\uncertaintyInequality"

use "proc_data\interestRate_longData.dta", clear 
keep HH_ID MONTH 
*Translate the month to month date type 
	cap drop month_date 
	generate tempDate= date(MONTH, "M Y")
	format tempDate %td
	g month= month(tempDate)
	g year = year(tempDate)
	gen modate=ym(year ,month)
	format modate %tm
	drop tempDate 

keep HH_ID MONTH modate
duplicates drop 
tempfile tempi
save `tempi'

forv month = 1/13 { //1/13{
	use "proc_data\interestRate_longData.dta", clear 
	drop index 

	*Translate the month to month date type 
		cap drop month_date 
		generate tempDate= date(MONTH, "M Y")
		format tempDate %td
		g month= month(tempDate)
		g year = year(tempDate)
		gen modate=ym(year ,month)
		format modate %tm
		drop tempDate 

	* Filter out missing data/different families 
		keep if REASON_FOR_NON_RESPONSE_x =="No Failure" & REASON_FOR_NON_RESPONSE_y == "No Failure" 
		keep if FAMILY_SHIFTED_x != "Y" & FAMILY_SHIFTED_y != "Y" 

	compress

	*Create the ID, which is HH_ID and the number of "consecutive runs" of dates 

	tsset HH_ID modate  
	cap drop  run maxrun 
	gen run = .
	by HH_ID: replace run = cond(L.run == ., 1, L.run + 1)
	by HH_ID: egen maxrun = max(run)
	g countRuns = run  ==1
	bysort HH_ID (modate) : gen runID = sum(countRuns )

	g double  HH_IDRun = HH_ID*10 + runID  //This is the unique ID for the FE
		
	*Create the cumlative sum variables for a run 
	bysort HH_IDRun (modate) : gen TotalReturns_cumsum = sum(TotalReturns)
	bysort HH_IDRun (modate) : gen TOTAL_EXPENDITURE_cumsum = sum(TOTAL_EXPENDITURE)
	bysort HH_IDRun (modate) : gen totalIncome_cumsum = sum(totalIncome)
	g growingCapital   = totalIncome_cumsum  + TotalReturns_cumsum  - TOTAL_EXPENDITURE_cumsum 

	*Create the lag variable 
	bys HH_IDRun : g lag_growingCapital  = growingCapital[_n-1]

	*Gen the wealth distribution 
		*Simple sanity test 
		reg TotalReturns lag_growingCapital  , r
	
	cap drop capital_estiamte capitalt_1 
	disp `month'
	reghdfe TotalReturns lag_growingCapital  if modate<ym(2019,`month'), abs(capital_estiamtes = HH_IDRun) vce(cluster HH_ID) nocons 
	g capitalt_`month'= capital_estiamtes/_b[lag_growingCapital]
	replace capitalt_`month'= capitalt_`month' + totalIncome_cumsum  + TotalReturns_cumsum  - TOTAL_EXPENDITURE_cumsum 

	keep HH_ID modate capitalt_`month'
	duplicates drop 

	merge 1:1 HH_ID modate using `tempi' , nogen
	save `tempi', replace 
	
}

export delimited using "proc_data\wealthDistribution.csv", replace
import delimited "proc_data\wealthDistribution.csv", clear 
cap drop modate
generate tempDate= date(month, "M Y")
format tempDate %td
g month2= month(tempDate)
g year = year(tempDate)
gen modate=ym(year ,month2)
format modate %tm
drop tempDate 


egen test = rowsd(capitalt_13 capitalt_12 capitalt_11 capitalt_10 capitalt_9 capitalt_8 capitalt_7 capitalt_6 capitalt_5 capitalt_4 capitalt_3 capitalt_2 capitalt_1)
*Explore the distribution at differenet month
sum capitalt_11 if modate == ym(2015,02) , d  
*Export
preserve
keep if modate == ym(2015,02)
keep  capitalt_11
replace capitalt_11 = capitalt_11/1000
outreg2 totalAssets using "presentation_outputs\figures_slides_11062022\Estimated_WealthDistribution.tex", sum(detail) eqdrop(N sum_w  Var skewness kurtosis sum) replace 
restore 

centile capitalt_6 if modate == ym(2015,02), centile(0(5)100) 




