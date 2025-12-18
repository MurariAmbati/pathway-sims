"""
ngf/trka pathway model
mathematical modeling of neurotrophin signaling cascades
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import pandas as pd
import json

class NGFTrkAModel:
    """
    comprehensive model of ngf/trka signaling pathways
    includes ras/mapk, pi3k/akt, and plcγ cascades
    """
    
    def __init__(self, ngf_concentration=10.0, trka_density=10000, 
                 pathways=None, feedback=True, crosstalk=True, noise=0.05):
        """
        initialize the signaling model
        
        parameters:
        - ngf_concentration: ngf ligand concentration (nm)
        - trka_density: trka receptors per cell
        - pathways: list of active pathways
        - feedback: include feedback loops
        - crosstalk: include pathway crosstalk
        - noise: biological noise level (0-1)
        """
        self.ngf = ngf_concentration
        self.trka_total = trka_density
        self.pathways = pathways or ["ras/mapk", "pi3k/akt", "plcγ"]
        self.feedback = feedback
        self.crosstalk = crosstalk
        self.noise = noise
        
        # kinetic parameters (rate constants)
        self.params = {
            # receptor binding and activation
            'k_bind': 0.1,          # ngf-trka binding (1/nm/min)
            'k_unbind': 0.01,       # ngf-trka dissociation (1/min)
            'k_dimer': 0.5,         # receptor dimerization (1/min)
            'k_auto': 0.8,          # autophosphorylation (1/min)
            'k_dephos_rec': 0.05,   # receptor dephosphorylation (1/min)
            
            # ras/mapk pathway
            'k_ras_act': 0.6,       # ras activation by grb2/sos (1/min)
            'k_ras_inact': 0.2,     # ras inactivation by gap (1/min)
            'k_raf_act': 0.7,       # raf activation by ras (1/min)
            'k_raf_inact': 0.15,    # raf inactivation (1/min)
            'k_mek_act': 0.8,       # mek phosphorylation by raf (1/min)
            'k_mek_inact': 0.12,    # mek dephosphorylation (1/min)
            'k_erk_act': 0.9,       # erk phosphorylation by mek (1/min)
            'k_erk_inact': 0.1,     # erk dephosphorylation (1/min)
            
            # pi3k/akt pathway
            'k_pi3k_act': 0.65,     # pi3k activation (1/min)
            'k_pi3k_inact': 0.18,   # pi3k inactivation (1/min)
            'k_pip3_form': 0.85,    # pip3 formation (1/min)
            'k_pip3_deg': 0.22,     # pip3 degradation by pten (1/min)
            'k_pdk1_act': 0.75,     # pdk1 activation (1/min)
            'k_akt_act': 0.8,       # akt phosphorylation (1/min)
            'k_akt_inact': 0.1,     # akt dephosphorylation (1/min)
            'k_bad_phos': 0.7,      # bad phosphorylation (1/min)
            'k_bad_dephos': 0.15,   # bad dephosphorylation (1/min)
            
            # plcγ pathway
            'k_plcg_act': 0.55,     # plcγ activation (1/min)
            'k_plcg_inact': 0.2,    # plcγ inactivation (1/min)
            'k_ip3_form': 0.8,      # ip3 formation (1/min)
            'k_ip3_deg': 0.3,       # ip3 degradation (1/min)
            'k_dag_form': 0.75,     # dag formation (1/min)
            'k_dag_deg': 0.25,      # dag degradation (1/min)
            'k_pkc_act': 0.6,       # pkc activation (1/min)
            'k_pkc_inact': 0.18,    # pkc inactivation (1/min)
            'k_ca_release': 0.9,    # calcium release (1/min)
            'k_ca_seq': 0.35,       # calcium sequestration (1/min)
            
            # feedback and crosstalk
            'k_fb_erk_sos': 0.3,    # erk negative feedback on sos
            'k_fb_akt_raf': 0.25,   # akt inhibition of raf
            'k_ct_pkc_raf': 0.2,    # pkc activation of raf
            'k_ct_ca_ras': 0.15,    # calcium modulation of ras
        }
        
        # initial conditions
        self.y0 = self._get_initial_conditions()
        
    def _get_initial_conditions(self):
        """set initial concentrations for all species"""
        return [
            self.ngf,           # 0: free ngf
            self.trka_total,    # 1: inactive trka
            0,                  # 2: ngf-trka complex
            0,                  # 3: trka dimer (active)
            1000,               # 4: inactive ras
            0,                  # 5: active ras-gtp
            500,                # 6: inactive raf
            0,                  # 7: active raf
            800,                # 8: inactive mek
            0,                  # 9: active mek
            1000,               # 10: inactive erk
            0,                  # 11: active erk
            600,                # 12: inactive pi3k
            0,                  # 13: active pi3k
            100,                # 14: pip2
            0,                  # 15: pip3
            0,                  # 16: active pdk1
            800,                # 17: inactive akt
            0,                  # 18: active akt
            500,                # 19: inactive bad
            0,                  # 20: phospho-bad (inactive)
            400,                # 21: inactive plcγ
            0,                  # 22: active plcγ
            0,                  # 23: ip3
            0,                  # 24: dag
            600,                # 25: inactive pkc
            0,                  # 26: active pkc
            100,                # 27: calcium (basal)
        ]
    
    def _ode_system(self, y, t):
        """
        system of ordinary differential equations
        describes temporal evolution of signaling network
        """
        p = self.params
        dydt = np.zeros(len(y))
        
        # unpack variables
        ngf, trka, ngf_trka, trka_dimer = y[0:4]
        ras_gdp, ras_gtp = y[4:6]
        raf, raf_a = y[6:8]
        mek, mek_a = y[8:10]
        erk, erk_a = y[10:12]
        pi3k, pi3k_a = y[12:14]
        pip2, pip3 = y[14:16]
        pdk1_a = y[16]
        akt, akt_a = y[17:19]
        bad, bad_p = y[19:21]
        plcg, plcg_a = y[21:23]
        ip3, dag = y[23:25]
        pkc, pkc_a = y[25:27]
        ca = y[27]
        
        # receptor activation
        dydt[0] = -p['k_bind'] * ngf * trka + p['k_unbind'] * ngf_trka
        dydt[1] = -p['k_bind'] * ngf * trka + p['k_unbind'] * ngf_trka + p['k_dephos_rec'] * trka_dimer
        dydt[2] = p['k_bind'] * ngf * trka - p['k_unbind'] * ngf_trka - p['k_dimer'] * ngf_trka
        dydt[3] = p['k_dimer'] * ngf_trka - p['k_dephos_rec'] * trka_dimer
        
        # ras/mapk pathway
        if "ras/mapk" in self.pathways:
            # feedback terms
            fb_erk = p['k_fb_erk_sos'] * erk_a if self.feedback else 0
            ct_ca = p['k_ct_ca_ras'] * (ca / 1000) if self.crosstalk else 0
            
            dydt[4] = -p['k_ras_act'] * ras_gdp * trka_dimer + p['k_ras_inact'] * ras_gtp + fb_erk * ras_gtp
            dydt[5] = p['k_ras_act'] * ras_gdp * trka_dimer - p['k_ras_inact'] * ras_gtp - fb_erk * ras_gtp + ct_ca * ras_gdp
            
            # crosstalk terms
            fb_akt = p['k_fb_akt_raf'] * akt_a if self.crosstalk else 0
            ct_pkc = p['k_ct_pkc_raf'] * pkc_a if self.crosstalk else 0
            
            dydt[6] = -p['k_raf_act'] * raf * ras_gtp + p['k_raf_inact'] * raf_a + fb_akt * raf_a
            dydt[7] = p['k_raf_act'] * raf * ras_gtp - p['k_raf_inact'] * raf_a - fb_akt * raf_a + ct_pkc * raf
            
            dydt[8] = -p['k_mek_act'] * mek * raf_a + p['k_mek_inact'] * mek_a
            dydt[9] = p['k_mek_act'] * mek * raf_a - p['k_mek_inact'] * mek_a
            
            dydt[10] = -p['k_erk_act'] * erk * mek_a + p['k_erk_inact'] * erk_a
            dydt[11] = p['k_erk_act'] * erk * mek_a - p['k_erk_inact'] * erk_a
        
        # pi3k/akt pathway
        if "pi3k/akt" in self.pathways:
            dydt[12] = -p['k_pi3k_act'] * pi3k * trka_dimer + p['k_pi3k_inact'] * pi3k_a
            dydt[13] = p['k_pi3k_act'] * pi3k * trka_dimer - p['k_pi3k_inact'] * pi3k_a
            
            dydt[14] = -p['k_pip3_form'] * pip2 * pi3k_a + p['k_pip3_deg'] * pip3
            dydt[15] = p['k_pip3_form'] * pip2 * pi3k_a - p['k_pip3_deg'] * pip3
            
            dydt[16] = p['k_pdk1_act'] * pip3 - 0.1 * pdk1_a
            
            dydt[17] = -p['k_akt_act'] * akt * pdk1_a + p['k_akt_inact'] * akt_a
            dydt[18] = p['k_akt_act'] * akt * pdk1_a - p['k_akt_inact'] * akt_a
            
            dydt[19] = -p['k_bad_phos'] * bad * akt_a + p['k_bad_dephos'] * bad_p
            dydt[20] = p['k_bad_phos'] * bad * akt_a - p['k_bad_dephos'] * bad_p
        
        # plcγ pathway
        if "plcγ" in self.pathways:
            dydt[21] = -p['k_plcg_act'] * plcg * trka_dimer + p['k_plcg_inact'] * plcg_a
            dydt[22] = p['k_plcg_act'] * plcg * trka_dimer - p['k_plcg_inact'] * plcg_a
            
            dydt[23] = p['k_ip3_form'] * pip2 * plcg_a - p['k_ip3_deg'] * ip3
            dydt[24] = p['k_dag_form'] * pip2 * plcg_a - p['k_dag_deg'] * dag
            
            dydt[25] = -p['k_pkc_act'] * pkc * dag * (ca / 100) + p['k_pkc_inact'] * pkc_a
            dydt[26] = p['k_pkc_act'] * pkc * dag * (ca / 100) - p['k_pkc_inact'] * pkc_a
            
            dydt[27] = p['k_ca_release'] * ip3 - p['k_ca_seq'] * ca
        
        # add biological noise
        if self.noise > 0:
            noise_vec = np.random.normal(0, self.noise, len(dydt))
            dydt = dydt + noise_vec * np.abs(dydt)
        
        return dydt
    
    def simulate(self, time_points):
        """
        run simulation and return results
        
        parameters:
        - time_points: array of time points for simulation
        
        returns:
        - dictionary of results for each species
        """
        solution = odeint(self._ode_system, self.y0, time_points)
        
        # package results
        results = {
            'ngf': solution[:, 0],
            'trka': solution[:, 1],
            'ngf_trka': solution[:, 2],
            'trka_dimer': solution[:, 3],
            'ras_gdp': solution[:, 4],
            'ras_gtp': solution[:, 5],
            'raf': solution[:, 6],
            'raf_active': solution[:, 7],
            'mek': solution[:, 8],
            'mek_active': solution[:, 9],
            'erk': solution[:, 10],
            'erk_active': solution[:, 11],
            'pi3k': solution[:, 12],
            'pi3k_active': solution[:, 13],
            'pip2': solution[:, 14],
            'pip3': solution[:, 15],
            'pdk1_active': solution[:, 16],
            'akt': solution[:, 17],
            'akt_active': solution[:, 18],
            'bad': solution[:, 19],
            'bad_phospho': solution[:, 20],
            'plcg': solution[:, 21],
            'plcg_active': solution[:, 22],
            'ip3': solution[:, 23],
            'dag': solution[:, 24],
            'pkc': solution[:, 25],
            'pkc_active': solution[:, 26],
            'calcium': solution[:, 27],
        }
        
        return results
    
    def calculate_survival_score(self):
        """calculate cell survival probability based on pathway activation"""
        if not hasattr(self, 'last_results'):
            return 0.0
        
        results = self.last_results
        
        # akt activation is primary survival signal
        akt_contrib = np.mean(results['akt_active']) / 800 * 70
        
        # bad phosphorylation (inactivation) promotes survival
        bad_contrib = np.mean(results['bad_phospho']) / 500 * 20
        
        # erk contributes to survival
        erk_contrib = np.mean(results['erk_active']) / 1000 * 10
        
        return min(100, akt_contrib + bad_contrib + erk_contrib)
    
    def calculate_differentiation_score(self):
        """calculate differentiation probability"""
        if not hasattr(self, 'last_results'):
            return 0.0
        
        results = self.last_results
        
        # mapk pathway is primary differentiation signal
        erk_contrib = np.mean(results['erk_active']) / 1000 * 60
        
        # calcium signaling contributes
        ca_contrib = (np.mean(results['calcium']) - 100) / 500 * 25
        
        # pkc activation
        pkc_contrib = np.mean(results['pkc_active']) / 600 * 15
        
        return min(100, erk_contrib + ca_contrib + pkc_contrib)
    
    def get_components_dataframe(self):
        """create dataframe of pathway components"""
        components = [
            {"component": "ngf", "type": "ligand", "pathway": "receptor"},
            {"component": "trka", "type": "receptor", "pathway": "receptor"},
            {"component": "ras", "type": "gtpase", "pathway": "ras/mapk"},
            {"component": "raf", "type": "kinase", "pathway": "ras/mapk"},
            {"component": "mek", "type": "kinase", "pathway": "ras/mapk"},
            {"component": "erk", "type": "kinase", "pathway": "ras/mapk"},
            {"component": "pi3k", "type": "kinase", "pathway": "pi3k/akt"},
            {"component": "pdk1", "type": "kinase", "pathway": "pi3k/akt"},
            {"component": "akt", "type": "kinase", "pathway": "pi3k/akt"},
            {"component": "bad", "type": "pro-apoptotic", "pathway": "pi3k/akt"},
            {"component": "plcγ", "type": "phospholipase", "pathway": "plcγ"},
            {"component": "pkc", "type": "kinase", "pathway": "plcγ"},
        ]
        return pd.DataFrame(components)
    
    def get_propagation_metrics(self):
        """calculate signal propagation metrics"""
        if not hasattr(self, 'last_results'):
            return pd.DataFrame()
        
        results = self.last_results
        
        metrics = []
        if "ras/mapk" in self.pathways:
            metrics.append({
                "pathway": "ras/mapk",
                "amplitude": f"{np.max(results['erk_active']):.1f}",
                "duration": "sustained",
                "latency": "2-5 min"
            })
        
        if "pi3k/akt" in self.pathways:
            metrics.append({
                "pathway": "pi3k/akt",
                "amplitude": f"{np.max(results['akt_active']):.1f}",
                "duration": "sustained",
                "latency": "1-3 min"
            })
        
        if "plcγ" in self.pathways:
            metrics.append({
                "pathway": "plcγ",
                "amplitude": f"{np.max(results['calcium']):.1f}",
                "duration": "transient",
                "latency": "0.5-2 min"
            })
        
        return pd.DataFrame(metrics)
    
    def export_to_csv(self, results, time_points):
        """export simulation results to csv format"""
        df = pd.DataFrame(results)
        df.insert(0, 'time', time_points)
        return df.to_csv(index=False)
    
    def export_to_json(self, results):
        """export simulation results to json format"""
        export_dict = {
            'parameters': {
                'ngf_concentration': self.ngf,
                'trka_density': self.trka_total,
                'pathways': self.pathways,
                'feedback': self.feedback,
                'crosstalk': self.crosstalk
            },
            'results': {k: v.tolist() for k, v in results.items()}
        }
        return json.dumps(export_dict, indent=2)
    
    def generate_report(self, results, time_points):
        """generate html report"""
        survival = self.calculate_survival_score()
        differentiation = self.calculate_differentiation_score()
        
        html = f"""
        <html>
        <head><title>ngf/trka simulation report</title></head>
        <body>
        <h1>neurotrophin signaling simulation</h1>
        <h2>parameters</h2>
        <p>ngf concentration: {self.ngf} nm</p>
        <p>trka density: {self.trka_total} receptors/cell</p>
        <p>simulation time: {time_points[-1]} minutes</p>
        <h2>outcomes</h2>
        <p>survival score: {survival:.1f}%</p>
        <p>differentiation score: {differentiation:.1f}%</p>
        </body>
        </html>
        """
        return html
    
    def create_dose_response_curve(self):
        """placeholder for dose-response visualization"""
        import plotly.graph_objects as go
        
        doses = np.logspace(-1, 2, 20)
        response = 100 * doses / (doses + 5)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=doses,
            y=response,
            mode='lines+markers',
            line=dict(color='#00d4ff', width=3)
        ))
        
        fig.update_layout(
            title="dose-response curve",
            xaxis_title="ngf concentration (nm)",
            yaxis_title="response (%)",
            xaxis_type="log",
            template="plotly_dark"
        )
        
        return fig
    
    def create_correlation_matrix(self, results):
        """create correlation matrix of pathway components"""
        import plotly.graph_objects as go
        
        # select key components
        keys = ['erk_active', 'akt_active', 'pkc_active', 'calcium']
        data = np.array([results[k] for k in keys if k in results])
        
        if len(data) == 0:
            return go.Figure()
        
        corr_matrix = np.corrcoef(data)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=keys[:len(data)],
            y=keys[:len(data)],
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="pathway correlation matrix",
            template="plotly_dark"
        )
        
        return fig
    
    def perform_sensitivity_analysis(self):
        """sensitivity analysis placeholder"""
        import plotly.graph_objects as go
        
        params = ['k_ras_act', 'k_akt_act', 'k_erk_act']
        sensitivity = np.random.uniform(0.5, 1.5, len(params))
        
        fig = go.Figure(data=[
            go.Bar(x=params, y=sensitivity, marker_color='#00d4ff')
        ])
        
        fig.update_layout(
            title="parameter sensitivity",
            yaxis_title="sensitivity index",
            template="plotly_dark"
        )
        
        return fig
    
    def create_bifurcation_diagram(self):
        """bifurcation analysis placeholder"""
        import plotly.graph_objects as go
        
        param_range = np.linspace(0.1, 2.0, 50)
        steady_states = 50 + 30 * np.sin(param_range * 3)
        
        fig = go.Figure(data=[
            go.Scatter(x=param_range, y=steady_states, mode='markers',
                      marker=dict(color='#00d4ff', size=3))
        ])
        
        fig.update_layout(
            title="bifurcation diagram",
            xaxis_title="parameter value",
            yaxis_title="steady state",
            template="plotly_dark"
        )
        
        return fig
    
    def analyze_stability(self):
        """stability analysis placeholder"""
        stability_data = [
            {"state": "low ngf", "type": "stable", "eigenvalue": "-0.15"},
            {"state": "high ngf", "type": "stable", "eigenvalue": "-0.08"},
            {"state": "oscillatory", "type": "unstable", "eigenvalue": "0.02 ± 0.3i"}
        ]
        return pd.DataFrame(stability_data)
