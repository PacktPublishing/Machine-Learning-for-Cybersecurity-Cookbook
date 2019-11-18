rule is_a_pdf 
{
       strings:
              $pdf_magic = {25 50 44 46}
       condition:
              $pdf_magic at 0
}
 
rule dummy_rule1
{
       condition:
              false
}
 
rule dummy_rule2
{
       condition:
              true
}