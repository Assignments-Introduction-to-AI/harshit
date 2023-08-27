import read, copy
from util import *
from logical_classes import *

verbose = 0

class KnowledgeBase(object):
    def __init__(self, facts=[], rules=[]):
        self.facts = facts
        self.rules = rules
        self.ie = InferenceEngine()

    def __repr__(self):
        return 'KnowledgeBase({!r}, {!r})'.format(self.facts, self.rules)

    def __str__(self):
        string = "Knowledge Base: \n"
        string += "\n".join((str(fact) for fact in self.facts)) + "\n"
        string += "\n".join((str(rule) for rule in self.rules))
        return string

    def _get_fact(self, fact):
        """INTERNAL USE ONLY
        Get the fact in the KB that is the same as the fact argument

        Args:
            fact (Fact): Fact we're searching for

        Returns:
            Fact: matching fact
        """
        for kbfact in self.facts:
            if fact == kbfact:
                return kbfact

    def _get_rule(self, rule):
        """INTERNAL USE ONLY
        Get the rule in the KB that is the same as the rule argument

        Args:
            rule (Rule): Rule we're searching for

        Returns:
            Rule: matching rule
        """
        for kbrule in self.rules:
            if rule == kbrule:
                return kbrule

    def kb_add(self, fact_rule):
        """Add a fact or rule to the KB
        Args:
            fact_rule (Fact or Rule) - Fact or Rule to be added
        Returns:
            None
        """
        printv("Adding {!r}", 1, verbose, [fact_rule])
        if isinstance(fact_rule, Fact):
            if fact_rule not in self.facts:
                self.facts.append(fact_rule)
                for rule in self.rules:
                    self.ie.fc_infer(fact_rule, rule, self)
            else:
                if fact_rule.supprt_by:
                    ind = self.facts.index(fact_rule)
                    for f in fact_rule.supprt_by:
                        self.facts[ind].supprt_by.append(f)
                else:
                    ind = self.facts.index(fact_rule)
                    self.facts[ind].asserted = True
        elif isinstance(fact_rule, Rule):
            if fact_rule not in self.rules:
                self.rules.append(fact_rule)
                for fact in self.facts:
                    self.ie.fc_infer(fact, fact_rule, self)
            else:
                if fact_rule.supprt_by:
                    ind = self.rules.index(fact_rule)
                    for f in fact_rule.supprt_by:
                        self.rules[ind].supprt_by.append(f)
                else:
                    ind = self.rules.index(fact_rule)
                    self.rules[ind].asserted = True

    def kb_assert(self, fact_rule):
        """Assert a fact or rule into the KB

        Args:
            fact_rule (Fact or Rule): Fact or Rule we're asserting
        """
        printv("Asserting {!r}", 0, verbose, [fact_rule])
        self.kb_add(fact_rule)

    def kb_ask(self, fact):
        """Ask if a fact is in the KB

        Args:
            fact (Fact) - Statement to be asked (will be converted into a Fact)

        Returns:
            listof Bindings|False - list of Bindings if result found, False otherwise
        """
        # print("Asking {!r}".format(fact))
        if factq(fact):
            f = Fact(fact.statement)
            bindings_lst = ListOfBindings()
            # ask matched facts
            for fact in self.facts:
                binding = match(f.statement, fact.statement)
                if binding:
                    bindings_lst.add_bindings(binding, [fact])

            return bindings_lst if bindings_lst.list_of_bindings else []

        else:
            # print("Invalid ask:", fact.statement)
            return []

    def kb_retract(self, fact_rule):
        """Retract a fact or a rule from the KB

        Args:
            fact_rule (Fact or Rule) - Fact or Rule to be retracted

        Returns:
            None
        """
        printv("Retracting {!r}", 0, verbose, [fact_rule])
        ####################################################
        # Student code goes here
        if isinstance(fact_rule, Fact) and fact_rule in self.facts:
            factindex = self.facts.index(fact_rule)
            factorrule = self.facts[factindex]
            if self.facts[factindex].asserted == True and not factorrule.supported_by:
                    
                for i in factorrule.supports_facts:
                    for j in i.supported_by:
                        if factorrule in j:
                            i.supported_by.remove(j)
                    self.kb_retract_helper(i)
                
                for i in factorrule.supports_rules:
                    for j in i.supported_by:
                        if factorrule in j:
                            i.supported_by.remove(j)
                    self.kb_retract_helper(i)
                
                self.facts.remove(self.facts[factindex])

    def kb_retract_helper(self, factorrule):
        if not factorrule: return
        if isinstance(factorrule, Fact):
            if (not factorrule.asserted) and (not factorrule.supported_by):

                for i in factorrule.supports_facts:
                    for j in i.supported_by:
                        if factorrule in j:
                            i.supported_by.remove(j)
                    self.kb_retract_helper(i)

                for i in factorrule.supports_rules:
                    for j in i.supported_by:
                        if factorrule in j:
                            i.supported_by.remove(j)
                    self.kb_retract_helper(i)

                self.facts.remove(factorrule)
                
        elif isinstance(factorrule, Rule):
            if (not factorrule.asserted) and (not factorrule.supported_by):

                for i in factorrule.supports_facts:
                    for j in i.supported_by:
                        if factorrule in j:
                            i.supported_by.remove(j)
                    self.kb_retract_helper(i)

                for i in factorrule.supports_rules:
                    for j in i.supported_by:
                        if factorrule in j:
                            i.supported_by.remove(j)
                    self.kb_retract_helper(i)

                self.rules.remove(factorrule)



class InferenceEngine(object):
    def fc_infer(self, fact, rule, kb):
        """Forward-chaining to infer new facts and rules

        Args:
            fact (Fact) - A fact from the KnowledgeBase
            rule (Rule) - A rule from the KnowledgeBase
            kb (KnowledgeBase) - A KnowledgeBase

        Returns:
            Nothing
        """
        printv('Attempting to infer from {!r} and {!r} => {!r}', 1, verbose,
            [fact.statement, rule.lhs, rule.rhs])
        ####################################################
        # Student code goes here

        binding = match(rule.lhs[0], fact.statement)
        a = [fact, rule]
        
        if binding:
            if len(rule.lhs) == 1:
                
                newfact = Fact(instantiate(rule.rhs, binding))
                
                if newfact not in kb.facts:
                    
                    newfact.asserted = False
                    newfact.supported_by.append(a)
                    kb.kb_add(newfact)
                
                    factindex = kb.facts.index(fact)
                    kb.facts[factindex].supports_facts.append(newfact)
                
                    ruleindex = kb.rules.index(rule)
                    kb.rules[ruleindex].supports_facts.append(newfact)
            
            elif len(rule.lhs) > 1:
                
                llhs = []
                
                for i in range(1,len(rule.lhs)):
                    add = instantiate(rule.lhs[i], binding)
                    llhs.append(add)
                
                newrule = Rule([llhs,instantiate(rule.rhs, binding)])
                newrule.supported_by.append(a)
                newrule.asserted = False
                
                fact.supports_rules.append(newrule)
                rule.supports_rules.append(newrule)
                kb.kb_add(newrule)
