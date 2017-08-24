// vim: set expandtab tabstop=4 shiftwidth=4 autoindent:
//
// File:   smartForm.js
// Mark Addinall - July 2017 
//
// Synopsis: This is a new application for Asset-IQ Dealer Network.
//           It replaces an existing PHP Web 1.0 type system that
//           has seved well enough but time for a replacement.
//
//           The following technologies are used in this project:
//
//
//           - HTML5
//           - CSS3
//           - ES6
//           - REACT
//           - REDUX
//           - jQuery
//           - Bootstrap
//           - Python
//           - Flask
//           - SQLAlchemy
//           - mySQL
//           - NPM
//           - Babel
//           - Webpack
//
//           The site will be built on a Client -> API -> Datastore architecture
//           Where the API is strictly defined.  Ad hoc access to data will not be
//           afforded.
//
//           SmartForm
//           this is a component that handles the rendering of ALL of the forms in the
//           application, from now until the end of time.  The calling container IMPORTS
//           a JSON object that is a DESCRIPTION of the form.  The 'SmartSuite'
//           of software developed in this application then looks after the rendering
//           of CHILD JSX code.
//
//           writing the parser was a LOT more difficult than I first thought.
//           problem being, JSX is not really a language, it is sugar for the BABEL
//           transpiler.  So what LOOKED reasonable as a stack based operation
//           in JSX was compiling to all sorts of illegal partial Javascript code.
//
//           Dereferencing the array of objects of arrays was (and is) a headache.
//
//           Something I didn't think of.  Given that ATTRIBUTES (in a form, these reference
//           input components of various sorts) may or may not ve visible, depending on the
//           public visibility set (the customer may choose to ignore, ie. 'not see'
//           a large number of the ATTRIBUTES, or, the end user may be logged in with
//           an application ROLE that does not allow the user to view the fields.
//
//           The dillema here is that the multiple pages are built BEFORE any of the FIELDS
//           are addressed.  I need to think about this one for a bit....
//

import React, { Component }                     from 'react'                                ;
import { connect }                              from 'react-redux'                          ;
import PropTypes                                from 'prop-types';
import { SmartRender }                          from './smartRender';
import { Accordion, AccordionSection }          from './accordion'      ;

import 'react-widgets/dist/css/react-widgets.css';

const pretend_props         = 'this.props';             // this is used as a special case in the
const endAccordionSection   = 'endAccordionSection';    // differentiate from the start of section
const endAccordion          = 'endAccordion';           // differentiate from the start of accordion

const singleOpen            = PropTypes.bool;           // prop for an Accordion
const openByDefault         = PropTypes.bool;           // prop for an AccordionSection
const uniqId                = PropTypes.string;         // prop for any type of form

const title                 = PropTypes.string;         // prop for any type of page
const noFunctions           = PropTypes.number;         // number of COMPONENTS in a form PAGE
const noFunctionsDone       = PropTypes.number;         // number of COMPONENTS TOUCHED in a form PAGE
const validated             = PropTypes.bool;           // component passes validation
const required              = PropTypes.bool;           // this can not be NULL

//---------------------------------
class SmartForm extends Component {

// this is a form description parser and renderer.
// it takes the description of a form in JSON format,
// passed in as a prop.


    //-----------------
    constructor(props) {

        super(props);
    }

    //-------------------
    formStart(formType) {

        // each page type in a single or multi page for will have it's own
        // format that has been decreed by the PROPS required by each
        // form initiation.

        let OP = {};

        console.log('formStart');
        switch(formType) {
            case 'accordion':
               // JSX.push(<Accordion uniqId={'addAsset'}{...this.props}singleOpen={true}>);
                OP.component = Accordion;
                OP.props = [uniqId, pretend_props, singleOpen];
                OP.data = ['addAsset', null, true];

                console.log('formStart = accordion');
                console.log(OP.component);
                break;
            case 'workflow':
                break;
            case 'modal':
                break;
            case 'normal':
                break;
        }   
        return(OP);
    }

    //-----------------
    formEnd(formType) {

        // each page type in a single or multi page for will have it's own
        // format that has been decreed by the PROPS required by each
        // form initiation.

        let OP = {};

        switch(formType) {
            case 'accordion':
                OP.component = Accordion;
                break;
            case 'workflow':
                break;
            case 'modal':
                break;
            case 'normal':
                break;
        }    
        return(OP);
    }

    //-------------------
    pageStart(pageType) {

        // each page type in a single or multi page for will have it's own
        // format that has been designed by the operations management
        // and the CSS designers.

        let OP = {};

        switch(pageType) {
            case 'accordion':
                       //JSX.push( <AccordionSection
                       // title=Asset Summary - Mandatories
                       // noFunctions=12
                       // noFunctionsDone=0
                       // validated={true}
                       // required={true}
                       // openByDefault={true} > );

                OP.component = AccordionSection;
                OP.props = [title, 
                            noFunctions, 
                            noFunctionsDone, 
                            validated, 
                            required, 
                            openByDefault];
                OP.data = ["Asset Summary - Mandatories", 
                            12, 0, true, true, true];
                break;
            case 'workflow':
                break;
            case 'modal':
                break;
            case 'normal':
                break;
            
        }
        return(OP);
    }
    
    //-----------------
    pageEnd(pageType) {

        // each page type in a single or multi page for will have it's own
        // format that has been designed by the operations management
        // and the CSS designers.

        let OP = {};

        switch(pageType) {
            case 'accordion':
                //JSX.push(</AccordionSection>);
                OP.component = AccordionSection;
                break;
            case 'workflow':
                break;
            case 'modal':
                break;
            case 'normal':
                break;
            
        }
        return(OP);
    }
    
    //-----------
    parse(form) {

        // this is the main form parser
        // the DESCRIPTION of the for comes into this
        // component as a JSON TREE
        // this parser traverses the tree and constructs the
        // JSX code to be rendered in the calling CONTAINER
        // these few lines of code handle ALL of the forms,
        // in ALL formats, in ALL of the application, from
        // now, until the end of days.

        let JSX = [];                                                       // container for the JSX to be rendered
        let element = '';                                                   // outside field element of form
        let pageElement = "";                                               // page elements
        let fieldElement = "";                                              // each of the field objects in the page
        let formFormat = '';                                                // snatch this value out of tree as we traverse

        console.log(form);

        for (element in form) {                                             // traverse the primary elements
            console.log('----------------------');
            console.log(element);
            switch(element) {                                               // which element are we on?
                case 'formName':                                            // we don't care about the name in here
                    break;
                case 'format':                                              // we DO care about the format!
                    console.log(form.format);
                    JSX.push(this.formStart(form.format));                  // handle the form intro JSX
                    break;
                case 'pages':
                    for (pageElement in form.pages) {                       // iterate for 1..N pages
                        JSX.push(this.pageStart(form.format));              // depending on the format, each new page gets unique JSX
                        console.log(form.pages[pageElement]);               // debugging
                        let pageArray = form.pages[pageElement];            // Pitman
                        for (fieldElement in pageArray.fields) {            // iterate down the LIST
                            console.log(fieldElement);                      // debugging
                            let fields = pageArray.fields[fieldElement];    // Pitman again
                            JSX.push(SmartRender(fields));                  // and process the field - this does a LOT of work as well!
                        }
                    JSX.push(this.pageEnd(form.format));                    // finish THIS page
                    }
            }
        }
        JSX.push(...this.formEnd(form.format));                             // finish off the rendering
        console.log(JSX);                                                   // debugging
        return(JSX);                                                        // send back the code for rendering
    }

    //--------
    render() {

        let code = this.parse(this.props.form);
        console.log("-----------------------");
        console.log("code from parse()");
        console.log("-----------------------");
        console.log(code);
        return(
                <div>
                    <h2>SmartForm Parser</h2>
                    {code.map(function(op, index) {
                        const CodeIndex = `${op.component}`;
                        console.log("-----------------------");
                        console.log("op from map()");
                        console.log("-----------------------");
                        console.log(op);
                        return(<div>
                                    <CodeIndex key={index} />
                               </div>
                        );
                    })}
                </div>
        );
    }
}
//-------------------------------------------------------------------------
export default SmartForm;

//-----------------   EOF -------------------------------------------------

