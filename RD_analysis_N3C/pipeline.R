

@transform_pandas(
    Output(rid="ri.vector.main.execute.4b03ad71-6ce9-4d32-8a0f-dd8f2f1f4957"),
    data_demo_comorb=Input(rid="ri.foundry.main.dataset.73c37aad-8f95-4bc3-8b30-0203f8b94ac6")
)
demo_comorb_char_table <- function(data_demo_comorb) {
  library(gtsummary)
  library(tidyverse)
  library(gt)

  trial <- as.data.frame(data_demo_comorb)
  trial %>% 
    tbl_summary(
      by = "rare_disease",
      digits = list(all_categorical() ~ c(0, 2), all_continuous() ~ 2)
    ) %>%
    add_p() %>%
    add_overall() %>%
    print()
}

@transform_pandas(
    Output(rid="ri.vector.main.execute.ba242f66-dacb-4bbc-8d12-06bf77722ea8"),
    univariate_hospitalization_rd=Input(rid="ri.vector.main.execute.11f11f84-6992-4a8a-86c4-e80b65127277")
)

forest_plot_RD_demographic_hospitalized <- function(univariate_hospitalization_rd) { 
   library(forestplot)
   library(dplyr)

    df = univariate_hospitalization_rd
    forestplot(
        labeltext = df$rds,
        mean = df$odds_ratio,
        upper = df$CI_upper,
        lower = df$CI_lower,
        pvalues = df$pvalue,
        zero = 1,
        title = "Hospitalized",
        xlab = "Odds Ratio", 
        clip = c(0.2,3), # specify the axis limits
    #    col = fpColors(box = 'blue', line = 'black, p='black')
        
    )
    
}

@transform_pandas(
    Output(rid="ri.vector.main.execute.8e7d5faa-89c5-467e-969f-2a8fea01761f"),
    univariate_death_rd=Input(rid="ri.vector.main.execute.19290d34-d521-427b-a149-495075c99d7d")
)

forest_plot_RD_demographic_mortality <- function(univariate_death_rd) { 
   library(forestplot)
   library(dplyr)

    df = univariate_death_rd
    forestplot(
        labeltext = df$rds,
        mean = df$odds_ratio,
        upper = df$CI_upper,
        lower = df$CI_lower,
        pvalues = df$pvalue,
        zero = 1,
        title = "Mortality",
        xlab = "Odds Ratio", 
        clip = c(0.2,3), # specify the axis limits
    #    col = fpColors(box = 'blue', line = 'black, p='black')
        
    )
    
}

@transform_pandas(
    Output(rid="ri.vector.main.execute.50ffabee-fa05-4662-9960-fc7d41ad0e1d"),
    hospitalized_model=Input(rid="ri.vector.main.execute.8575db2c-e0bd-468c-ae90-b92d0f02320e")
)
   

forest_plot_RDs_hospitalized <- function(hospitalized_model) { 
   library(forestplot)
   library(dplyr)

    df = hospitalized_model
    # Remove underscores and capitalize the first letter
    labeltext_modified <- gsub("_", " ", df$rds)
    labeltext_modified <- tools::toTitleCase(labeltext_modified)

    forestplot(
        labeltext = labeltext_modified,
        mean = df$odds_ratio,
        upper = df$CI_upper,
        lower = df$CI_lower,
        pvalues = df$pvalue,
        zero = 1,
        title = "Hospitalized",
        xlab = "Odds Ratio", 
        clip = c(0.9, 6),
        # Increase font size for axis labels and tick labels
        tsize = 10,  # Adjust as needed
        txt_gp = fpTxtGp(cex= 1.05),
        xticks = c(0.5, 1, 2, 3, 4, 5, 6),  # Customize x-axis tick labels
        yaxis.space = unit(0.5, "in"),  # Add space between y-axis label and figure
        xtickfontsize = 30  # Increase font size for x-axis tick labels
    )
}

@transform_pandas(
    Output(rid="ri.vector.main.execute.f641e04b-ba00-4427-8eb0-534d374395ed"),
    mortality_model=Input(rid="ri.vector.main.execute.4ed109de-5d14-4c29-9a4b-0059c78d90ce")
)
forest_plot_RDs_mortality <- function(mortality_model) { 
   library(forestplot)
   library(dplyr)

    df = mortality_model
    # Remove underscores and capitalize the first letter
    labeltext_modified <- gsub("_", " ", df$rds)
    labeltext_modified <- tools::toTitleCase(labeltext_modified)

    forestplot(
        labeltext = labeltext_modified,
        mean = df$odds_ratio,
        upper = df$CI_upper,
        lower = df$CI_lower,
        pvalues = df$pvalue,
        zero = 1,
        title = "Mortality",
        xlab = "Odds Ratio", 
        clip = c(0.9, 6.5),
        # Increase font size for axis labels and tick labels
        tsize = 10,  # Adjust as needed
        txt_gp = fpTxtGp(cex= 1.05),
        xticks = c(0.5, 1, 2, 3, 4, 5, 6, 7),  # Customize x-axis tick labels
        yaxis.space = unit(0.5, "in"),  # Add space between y-axis label and figure
        xtickfontsize = 30  # Increase font size for x-axis tick labels
    )
}

